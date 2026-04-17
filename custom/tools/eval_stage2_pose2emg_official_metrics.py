from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# File path:
#   .../golf_third_party/musclesinaction/custom/tools/eval_stage2_pose2emg_official_metrics.py
_MIA_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.models.frame_codebook import FrameCodebookConfig, FrameCodebookModel, ModalityConfig  # noqa: E402
from custom.models.temporal import TemporalConfig  # noqa: E402
from custom.models.vq_ema import VQEMAConfig  # noqa: E402
from custom.stage2.models.stage2_pose2emg import Stage2Pose2EMG, Stage2Pose2EMGConfig  # noqa: E402
from custom.stage2.models.dcsa import DCSAConfig  # noqa: E402
from custom.stage2.models.dstformer import DSTFormerConfig  # noqa: E402
from custom.stage2.models.fusion import ResidualAddConfig  # noqa: E402
from custom.stage2.models.temporal_backbone import TCNBackboneConfig  # noqa: E402
from custom.utils.mia_filelist import build_mia_train_filelist  # noqa: E402
from custom.utils.online_standardize import OnlineStandardizeConfig  # noqa: E402
from custom.utils.path_utils import get_musclesinaction_repo_root  # noqa: E402


MUSCLE_NAMES: Tuple[str, ...] = (
    "rightquad",
    "leftquad",
    "rightham",
    "leftham",
    "rightglutt",
    "leftglutt",
    "leftbicep",
    "rightbicep",
)


class _NullLogger:
    def info(self, *args, **kwargs):
        return

    def warning(self, *args, **kwargs):
        return

    def exception(self, *args, **kwargs):
        return


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_split(requested_split: str, mia_root: Path) -> str:
    req = str(requested_split).strip().lower()
    if req == "test":
        test_dir = mia_root / "MIADatasetOfficial" / "test"
        if test_dir.exists():
            return "test"
        print("[Warn] MIADatasetOfficial/test 不存在，回退到 val。")
        return "val"
    if req not in ("train", "val"):
        raise ValueError(f"Unsupported split: {requested_split}")
    return req


def _write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _get_filepaths_from_batch(batch: Dict[str, Any], batch_size: int) -> List[str]:
    fps = batch.get("filepath", None)
    if isinstance(fps, (list, tuple)):
        return [str(x) for x in fps]
    if isinstance(fps, str):
        return [fps] * batch_size
    return [f"sample_{i:06d}" for i in range(batch_size)]


def _root_center_joints3d(joints3d: torch.Tensor, root_index: int) -> torch.Tensor:
    root = joints3d[:, :, root_index : root_index + 1, :]
    return joints3d - root


def _emg_standardizer_std(standardizer) -> torch.Tensor:
    """与 OnlineStandardizer 一致的安全 std，用于反归一化。"""
    var_safe = torch.clamp(standardizer.var, min=1e-4)
    return torch.sqrt(var_safe + 1e-6)


def _emg_standardizer_stats_bt8(standardizer, *, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = standardizer.mean.to(device)
    std = _emg_standardizer_std(standardizer).to(device)
    m = int(mean.numel())
    if m == 8:
        mean_bt8 = mean.view(1, 1, 8).expand(1, int(t), 8)
        std_bt8 = std.view(1, 1, 8).expand(1, int(t), 8)
        return mean_bt8, std_bt8
    if m == int(t) * 8:
        mean_bt8 = mean.view(1, int(t), 8)
        std_bt8 = std.view(1, int(t), 8)
        return mean_bt8, std_bt8
    raise ValueError(f"Unsupported EMG standardizer dim={m}; expected 8 or T*8={int(t)*8}")


def _compute_official_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")
    if pred.ndim != 2 or pred.shape[1] != 8:
        raise ValueError(f"Expected flattened EMG shape (N,8), got {pred.shape}")

    sq_err = np.square(pred - gt)
    abs_err = np.abs(pred - gt)
    overall_mse = float(np.mean(sq_err))
    overall_rmse = float(math.sqrt(overall_mse))
    overall_mae = float(np.mean(abs_err))

    channel_mse = np.mean(sq_err, axis=0)
    channel_rmse = np.sqrt(channel_mse)
    channel_mae = np.mean(abs_err, axis=0)
    per_channel = []
    for idx, name in enumerate(MUSCLE_NAMES):
        per_channel.append(
            {
                "index": idx,
                "name": name,
                "mse": float(channel_mse[idx]),
                "rmse": float(channel_rmse[idx]),
                "mae": float(channel_mae[idx]),
            }
        )

    return {
        "official_global_mse": overall_mse,
        "official_global_rmse": overall_rmse,
        "official_global_mae": overall_mae,
        "num_frames": int(pred.shape[0]),
        "num_values": int(pred.size),
        "per_channel": per_channel,
    }


def _summarize_code_usage(
    *,
    name: str,
    global_counts: np.ndarray,
    batch_unique_counts: Sequence[int],
    batch_perplexities: Sequence[float],
    num_codes: int,
) -> Dict[str, Any]:
    global_counts = np.asarray(global_counts, dtype=np.int64).reshape(-1)
    active_mask = global_counts > 0
    active_codes = int(active_mask.sum())
    total_assignments = int(global_counts.sum())
    active_rate = float(active_codes / max(int(num_codes), 1))
    probs = global_counts.astype(np.float64) / max(total_assignments, 1)
    used_probs = probs[probs > 0]
    global_perplexity = float(np.exp(-np.sum(used_probs * np.log(used_probs)))) if used_probs.size > 0 else 0.0

    top_idx = np.argsort(global_counts)[::-1][: min(10, int(num_codes))]
    top_codes = [
        {
            "code": int(i),
            "count": int(global_counts[i]),
            "fraction": float(global_counts[i] / max(total_assignments, 1)),
        }
        for i in top_idx
        if int(global_counts[i]) > 0
    ]

    return {
        "name": name,
        "num_codes": int(num_codes),
        "active_codes": active_codes,
        "active_rate": active_rate,
        "total_assignments": total_assignments,
        "avg_batch_unique_codes": float(np.mean(batch_unique_counts)) if batch_unique_counts else 0.0,
        "avg_batch_active_rate": float(np.mean(batch_unique_counts) / max(int(num_codes), 1)) if batch_unique_counts else 0.0,
        "min_batch_unique_codes": int(np.min(batch_unique_counts)) if batch_unique_counts else 0,
        "max_batch_unique_codes": int(np.max(batch_unique_counts)) if batch_unique_counts else 0,
        "avg_batch_perplexity": float(np.mean(batch_perplexities)) if batch_perplexities else 0.0,
        "global_assignment_perplexity": global_perplexity,
        "top_codes": top_codes,
        "histogram": global_counts.astype(int).tolist(),
    }


def _build_stage1_from_ckpt(ckpt_path: Path, device: torch.device) -> FrameCodebookModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or "config" not in payload or "model_state" not in payload:
        raise RuntimeError(f"Stage1 checkpoint missing keys (config/model_state): {ckpt_path}")
    cfg = payload["config"]

    model_cfg = cfg["model"]
    vq_kwargs = dict(model_cfg["vq"])
    if "beta" not in vq_kwargs and "commitment_weight" in vq_kwargs:
        vq_kwargs["beta"] = vq_kwargs.pop("commitment_weight")
    vq_cfg = VQEMAConfig(**vq_kwargs)
    mods = model_cfg["modalities"]

    def _mod(name: str, d: Dict[str, Any]) -> ModalityConfig:
        std_cfg = OnlineStandardizeConfig(**(d.get("std", {}) or {}))
        temporal_cfg = TemporalConfig(**(d.get("temporal", {}) or {}))
        return ModalityConfig(
            name=name,
            in_dim=int(d["in_dim"]),
            hidden_dim=int(d["hidden_dim"]),
            token_count=int(d["token_count"]),
            code_dim=int(d["code_dim"]),
            encoder_type=str(d.get("encoder_type", "mixer")),
            decoder_type=str(d.get("decoder_type", "mixer")),
            recon_weight=float(d.get("recon_weight", 1.0)),
            online_std=bool(d.get("online_std", True)),
            std_cfg=std_cfg,
            temporal=temporal_cfg,
            mixer=d.get("mixer"),
            frame_mixer=d.get("frame_mixer"),
        )

    stage1_cfg = FrameCodebookConfig(
        vq=vq_cfg,
        joints3d=_mod("joints3d", mods["joints3d"]),
        smpl_pose=None,
        emg=_mod("emg", mods["emg"]),
        encoder_decoder_only=bool(model_cfg.get("encoder_decoder_only", False)),
    )
    m = FrameCodebookModel(stage1_cfg).to(device)
    m.load_state_dict(payload["model_state"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _build_stage2_from_ckpt(
    stage2_ckpt: Path, device: torch.device, *, stage1_override_ckpt: Path | None
) -> Tuple[Stage2Pose2EMG, Dict[str, Any], Path, FrameCodebookModel]:
    payload = torch.load(stage2_ckpt, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload or "config" not in payload:
        raise RuntimeError(f"Stage2 checkpoint missing keys (config/model_state): {stage2_ckpt}")
    cfg = payload["config"]

    stage1_ckpt_str = None
    if stage1_override_ckpt is not None:
        stage1_ckpt_str = str(stage1_override_ckpt)
    else:
        stage1_ckpt_str = str(payload.get("stage1_checkpoint", cfg.get("stage1", {}).get("checkpoint", "")))
    if not stage1_ckpt_str:
        raise ValueError("Missing stage1 checkpoint: provide cfg.stage1.checkpoint or store in stage2 checkpoint payload.")

    mia_root = get_musclesinaction_repo_root()
    stage1_path = (mia_root / stage1_ckpt_str).resolve() if not Path(stage1_ckpt_str).is_absolute() else Path(stage1_ckpt_str)
    stage1 = _build_stage1_from_ckpt(stage1_path, device=device)

    mcfg = cfg["model"]
    fusion_type = str(mcfg.get("fusion_type", "dcsa")).strip().lower()
    temporal_type = str(mcfg.get("temporal_type", "dstformer")).strip().lower()
    dim = int(mcfg.get("dim", 256))
    residual_add_cfg = None
    if fusion_type in ("residual_add", "residual", "add"):
        _r = dict(mcfg.get("fusion_residual_add", {}) or {})
        _r["dim"] = dim
        residual_add_cfg = ResidualAddConfig(**_r)
    tcn_cfg = None
    if temporal_type == "tcn":
        _t = dict(mcfg.get("tcn", {}) or {})
        _t["dim"] = dim
        tcn_cfg = TCNBackboneConfig(**_t)

    stage2_cfg = Stage2Pose2EMGConfig(
        token_count=int(mcfg.get("token_count", 63)),
        dim=dim,
        cont_encoder_type=str(mcfg.get("cont_encoder_type", "mixer")).strip().lower(),
        cont_hidden_dim=int(mcfg.get("cont_hidden_dim", 1024)),
        cont_joint_hidden_dim=int(mcfg.get("cont_joint_hidden_dim", 128)),
        fusion_type=fusion_type,
        dcsa=DCSAConfig(**(mcfg.get("dcsa", {}) or {})),
        fusion_residual_add=residual_add_cfg,
        temporal_type=temporal_type,
        dst=DSTFormerConfig(**(mcfg.get("dst", {}) or {})),
        tcn=tcn_cfg,
        emg_head_type=str(mcfg.get("emg_head_type", "mixer")).strip().lower(),
        emg_hidden=int(mcfg.get("emg_hidden", 256)),
        emg_mixer_hidden_dim=int(mcfg.get("emg_mixer_hidden_dim", 256)),
        emg_mixer_num_layers=int(mcfg.get("emg_mixer_num_layers", 4)),
        emg_pred_mode=str(mcfg.get("emg_pred_mode", "full")).strip().lower(),
        max_seq_len=int(mcfg.get("max_seq_len", 256)),
    )
    model = Stage2Pose2EMG(stage2_cfg, stage1=stage1).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, cfg, stage1_path, stage1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="custom/configs/stage2_pose2emg_eval.yaml",
        help="Eval config yaml for stage2 pipeline.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of samples to evaluate (overrides config data.max_samples). None = use all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda:0, cuda:6). Default from config runtime.device.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    runtime = cfg.get("runtime", {})
    data_cfg = cfg.get("data", {})
    stage2_cfg = cfg.get("stage2", {})
    out_cfg = cfg.get("output", {})

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if (str(runtime.get("device", "cuda")) == "cuda" and torch.cuda.is_available()) else "cpu")

    split = _resolve_split(str(data_cfg.get("split", "val")), mia_root)
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(runtime.get("num_workers", 4))
    max_samples = args.max_samples if getattr(args, "max_samples", None) is not None else data_cfg.get("max_samples", None)
    require_files = data_cfg.get("require_files", ["emgvalues.npy", "joints3d.npy"])
    # 未显式配置 filelist 时按 split 用 miaofficial_{split}_eval.txt，避免 split 改 train 却仍用 val 列表
    filelist_str = data_cfg.get("filelist")
    if not filelist_str:
        filelist_str = f"custom/tools/datasetsplits/miaofficial_{split}_eval.txt"
    filelist = Path(str(filelist_str))
    if not filelist.is_absolute():
        filelist = (mia_root / filelist).resolve()
    if not filelist.exists() or filelist.stat().st_size == 0:
        res = build_mia_train_filelist(
            mia_repo_root=mia_root,
            split=split,
            out_txt=filelist,
            max_samples=max_samples,
            require_files=list(require_files),
        )
        print(f"[FileList] wrote {res.num_samples} samples -> {res.filelist_path}")

    # dataset
    from musclesinaction.dataloader.data import MyMuscleDataset, _seed_worker  # type: ignore

    dset = MyMuscleDataset(
        str(filelist),
        _NullLogger(),
        split,
        percent=float(data_cfg.get("percent", 1.0)),
        step=int(data_cfg.get("step", 30)),
        std=str(data_cfg.get("std", "False")),
        cond=str(data_cfg.get("cond", "True")),
        transform=None,
    )
    if max_samples is not None:
        max_n = min(len(dset), int(max_samples))
        dset = Subset(dset, range(max_n))
        print(f"[EvalStage2] max_samples={max_samples} -> evaluating {max_n} samples")
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        worker_init_fn=_seed_worker,
        pin_memory=(device.type == "cuda"),
    )

    stage2_ckpt = Path(str(stage2_cfg.get("checkpoint", "")))
    if not stage2_ckpt.is_absolute():
        stage2_ckpt = (mia_root / stage2_ckpt).resolve()
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found: {stage2_ckpt}")

    stage1_override = cfg.get("stage1", {}).get("checkpoint", None)
    stage1_override_path = None
    if stage1_override:
        p = Path(str(stage1_override))
        stage1_override_path = p if p.is_absolute() else (mia_root / p).resolve()

    model, train_cfg, stage1_path, stage1 = _build_stage2_from_ckpt(stage2_ckpt, device, stage1_override_ckpt=stage1_override_path)

    joints3d_root_center = bool(data_cfg.get("joints3d_root_center", True))
    joints3d_root_index = int(data_cfg.get("joints3d_root_index", 8))

    # 训练时若用 Stage1 Standardizer 归一化 target（full + emg_normalize_target），推理输出在归一化空间，需反归一化后再与 raw gt 算指标
    pred_mode = str(train_cfg.get("model", {}).get("emg_pred_mode", "full")).strip().lower()
    emg_normalize_target = bool(train_cfg.get("data", {}).get("emg_normalize_target", False))
    stage1_emg = getattr(stage1, "emg", None)
    emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
    use_emg_denorm = (
        pred_mode == "full"
        and emg_normalize_target
        and emg_standardizer is not None
    )
    if use_emg_denorm:
        print("[EvalStage2] EMG 预测在归一化空间，将用 Stage1 Standardizer 反归一化后与 raw GT 算 RMSE/MAE")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_cfg.get("out_dir", None)
    if out_dir:
        out_root = Path(str(out_dir))
        out_root = out_root if out_root.is_absolute() else (mia_root / out_root).resolve()
    else:
        out_root = (mia_root / "custom" / "output" / "eval_stage2_pose2emg_official" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / stage2_ckpt.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sequence_csv = run_dir / "per_sequence.csv"
    summary_json = run_dir / "summary.json"
    pred_npz = run_dir / "pred_vs_gt.npz"
    codebook_json = run_dir / "codebook_usage.json"

    print("[EvalStage2] split =", split)
    print("[EvalStage2] device =", device)
    print("[EvalStage2] stage2_ckpt =", stage2_ckpt)
    print("[EvalStage2] stage1_ckpt =", stage1_path)
    print("[EvalStage2] filelist =", filelist)
    print("[EvalStage2] out_root =", out_root)

    num_codes = int(stage1.cfg.vq.num_codes)
    j3d_global_counts = np.zeros(num_codes, dtype=np.int64)
    emg_global_counts = np.zeros(num_codes, dtype=np.int64)
    j3d_batch_unique_counts: List[int] = []
    emg_batch_unique_counts: List[int] = []
    j3d_batch_perplexities: List[float] = []
    emg_batch_perplexities: List[float] = []

    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []

    sample_counter = 0
    batch_counter = 0

    for batch in tqdm(loader, desc="eval", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)  # (B,T,25,3)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)  # (B,8,T)
        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()  # (B,T,8)

        with torch.no_grad():
            out = model(joints3d)
            pred_bt8 = out["emg_pred"]  # (B,T,8)，训练用归一化 target 时此处为归一化空间
            if use_emg_denorm:
                mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(pred_bt8.shape[1]), device=device)
                pred_bt8 = pred_bt8 * std_bt8 + mean_bt8
            idx_j = out["idx_j3d"]  # (B,T,63)

            # stage1 EMG code usage over GT EMG (for consistent "emg usage" stats)
            b, t, _ = gt_bt8.shape
            emg_in_dim = int(stage1.emg.cfg.in_dim)
            if emg_in_dim == 8:
                emg_flat = gt_bt8.reshape(b * t, 8)
                _, _, idx_e, _, ppl_e, _ = stage1._encode_quantize(stage1.emg, emg_flat)  # type: ignore
            else:
                emg_clip = gt_bt8.reshape(b, t * 8)
                if int(emg_clip.shape[1]) != emg_in_dim:
                    raise ValueError(f"Stage1 emg.in_dim={emg_in_dim} but got gt clip dim={int(emg_clip.shape[1])}")
                _, _, idx_e, _, ppl_e, _ = stage1._encode_quantize(stage1.emg, emg_clip)  # type: ignore

        pred_np = pred_bt8.detach().cpu().numpy()
        gt_np = gt_bt8.detach().cpu().numpy()
        pred_all.append(pred_np.reshape(pred_np.shape[0] * pred_np.shape[1], 8))
        gt_all.append(gt_np.reshape(gt_np.shape[0] * gt_np.shape[1], 8))

        # code usage stats
        idx_j_np = idx_j.detach().cpu().numpy().reshape(-1)
        idx_e_np = idx_e.detach().cpu().numpy().reshape(-1)
        j3d_global_counts += np.bincount(idx_j_np, minlength=num_codes)
        emg_global_counts += np.bincount(idx_e_np, minlength=num_codes)
        j3d_batch_unique_counts.append(int(np.unique(idx_j_np).size))
        emg_batch_unique_counts.append(int(np.unique(idx_e_np).size))

        # batch perplexity from assignments
        def _assignment_ppl(ids: np.ndarray) -> float:
            cnt = np.bincount(ids, minlength=num_codes).astype(np.float64)
            p = cnt / max(float(cnt.sum()), 1.0)
            p = p[p > 0]
            return float(np.exp(-np.sum(p * np.log(p)))) if p.size > 0 else 0.0

        j3d_batch_perplexities.append(_assignment_ppl(idx_j_np))
        emg_batch_perplexities.append(float(ppl_e.item()))

        # per-sequence metrics
        filepaths = _get_filepaths_from_batch(batch, int(gt_np.shape[0]))
        for i in range(int(gt_np.shape[0])):
            seq_sq_err = np.square(pred_np[i] - gt_np[i])
            seq_abs_err = np.abs(pred_np[i] - gt_np[i])
            row = {
                "eval_index": sample_counter,
                "filepath": filepaths[i],
                "frames": int(gt_np.shape[1]),
                "mse": float(np.mean(seq_sq_err)),
                "rmse": float(math.sqrt(np.mean(seq_sq_err))),
                "mae": float(np.mean(seq_abs_err)),
            }
            for ch in range(8):
                row[f"rmse_{MUSCLE_NAMES[ch]}"] = float(math.sqrt(np.mean(seq_sq_err[:, ch])))
            per_sequence_rows.append(row)
            sample_counter += 1

        batch_counter += 1

    pred_cat = np.concatenate(pred_all, axis=0)
    gt_cat = np.concatenate(gt_all, axis=0)
    metrics = _compute_official_metrics(pred_cat, gt_cat)

    codebook_stats = {
        "joints3d": _summarize_code_usage(
            name="joints3d",
            global_counts=j3d_global_counts,
            batch_unique_counts=j3d_batch_unique_counts,
            batch_perplexities=j3d_batch_perplexities,
            num_codes=num_codes,
        ),
        "emg": _summarize_code_usage(
            name="emg",
            global_counts=emg_global_counts,
            batch_unique_counts=emg_batch_unique_counts,
            batch_perplexities=emg_batch_perplexities,
            num_codes=num_codes,
        ),
    }

    per_sequence_fields = [
        "eval_index",
        "filepath",
        "frames",
        "mse",
        "rmse",
        "mae",
        *[f"rmse_{name}" for name in MUSCLE_NAMES],
    ]
    _write_csv_rows(per_sequence_csv, per_sequence_rows, per_sequence_fields)

    summary = {
        "config_path": str(cfg_path),
        "split": split,
        "filelist": str(filelist),
        "stage2_checkpoint": str(stage2_ckpt),
        "stage1_checkpoint": str(stage1_path),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_samples": max_samples,
        "num_batches": int(batch_counter),
        "num_sequences": int(len(per_sequence_rows)),
        **metrics,
        "codebook_usage": codebook_stats,
        "outputs": {
            "per_sequence_csv": str(per_sequence_csv),
            "pred_vs_gt_npz": str(pred_npz),
            "codebook_usage_json": str(codebook_json),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(codebook_json, "w", encoding="utf-8") as f:
        json.dump(codebook_stats, f, ensure_ascii=False, indent=2)
    np.savez_compressed(pred_npz, pred_emg=pred_cat.astype(np.float32), gt_emg=gt_cat.astype(np.float32))

    print(
        "[EvalStage2Result]",
        {
            "stage2_ckpt": stage2_ckpt.name,
            "official_global_rmse": summary["official_global_rmse"],
            "official_global_mse": summary["official_global_mse"],
            "num_sequences": summary["num_sequences"],
            "j3d_active_codes": summary["codebook_usage"]["joints3d"]["active_codes"],
            "j3d_active_rate": summary["codebook_usage"]["joints3d"]["active_rate"],
        },
    )
    print("[EvalStage2] saved_summary_json =", summary_json)


if __name__ == "__main__":
    main()
