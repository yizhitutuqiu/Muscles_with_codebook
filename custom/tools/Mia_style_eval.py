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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_MIA_CUSTOM_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_CUSTOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_CUSTOM_ROOT))

from custom.models.frame_codebook import FrameCodebookConfig, FrameCodebookModel, ModalityConfig  # noqa: E402
from custom.models.temporal import TemporalConfig  # noqa: E402
from custom.models.vq_ema import VQEMAConfig  # noqa: E402
from custom.stage2.models.dcsa import DCSAConfig  # noqa: E402
from custom.stage2.models.dstformer import DSTFormerConfig  # noqa: E402
from custom.stage2.models.fusion import ResidualAddConfig  # noqa: E402
from custom.stage2.models.stage2_pose2emg import Stage2Pose2EMG, Stage2Pose2EMGConfig  # noqa: E402
from custom.stage2.models.temporal_backbone import TCNBackboneConfig  # noqa: E402
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


def _emg_standardizer_std(standardizer) -> torch.Tensor:
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
    # Unified-clip Stage1 standardizer: m == clip_len*8 (e.g. clip_len=5 -> 40).
    # Repeat stats across time to match current Stage2 sequence length T.
    if m % 8 == 0:
        clip_len = m // 8
        if clip_len > 0 and int(t) % int(clip_len) == 0:
            mean_clip = mean.view(1, int(clip_len), 8).repeat(1, int(t) // int(clip_len), 1)
            std_clip = std.view(1, int(clip_len), 8).repeat(1, int(t) // int(clip_len), 1)
            return mean_clip, std_clip
    raise ValueError(f"Unsupported EMG standardizer dim={m}; expected 8 or T*8={int(t)*8}")


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
    stage2_ckpt: Path, device: torch.device, *, stage1_override_ckpt: Optional[Path]
) -> Tuple[Stage2Pose2EMG, Dict[str, Any], Path, FrameCodebookModel]:
    payload = torch.load(stage2_ckpt, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload or "config" not in payload:
        raise RuntimeError(f"Stage2 checkpoint missing keys (config/model_state): {stage2_ckpt}")
    cfg = payload["config"]

    if stage1_override_ckpt is not None:
        stage1_path = stage1_override_ckpt
    else:
        stage1_ckpt_str = str(payload.get("stage1_checkpoint", cfg.get("stage1", {}).get("checkpoint", "")))
        if not stage1_ckpt_str:
            raise ValueError("Missing stage1 checkpoint: provide methods.stage2.stage1_checkpoint or store in stage2 checkpoint payload.")
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
        use_cond=bool(mcfg.get("use_cond", False)),
    )
    model = Stage2Pose2EMG(stage2_cfg, stage1=stage1).to(device)
    # The new clip-unified checkpoint might contain 'model.stage1.emg.temporal.pos_embed' weights 
    # but the stage1 config doesn't have it if it wasn't re-saved. Use strict=False to bypass.
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()
    return model, cfg, stage1_path, stage1


def _load_official_model(ckpt_path: Path, device: torch.device):
    from musclesinaction.models.modelposetoemg import TransformerEnc  # type: ignore

    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(payload)}")

    if "model_args" in payload:
        model_args = dict(payload["model_args"])
    else:
        model_args = dict(
            threed="True",
            num_tokens=1,
            dim_model=128,
            num_classes=8,
            num_heads=8,
            classif=False,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dropout_p=0.1,
            device=str(device),
            embedding="False",
            step=30,
        )
    model_args["device"] = str(device)
    model_args["step"] = int(model_args.get("step", 30))
    if str(model_args.get("threed", "True")) != "True":
        raise ValueError(f"This evaluator expects threed=True official model, got {model_args.get('threed')}")
    if int(model_args["step"]) != 30:
        raise ValueError(f"Official evaluator currently assumes step=30, got {model_args['step']}")

    model = TransformerEnc(**model_args).to(device)
    state = payload.get("my_model", None)
    if state is None:
        state = {k: v for k, v in payload.items() if isinstance(k, str)}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload, model_args


@dataclass(frozen=True)
class EvalCase:
    protocol: str
    train_filelist: Path
    val_filelist: Path
    split: str


def _discover_eval_cases(cfg: Dict[str, Any]) -> List[EvalCase]:
    protocols = cfg.get("protocols", []) or []
    cases: List[EvalCase] = []
    mia_root = get_musclesinaction_repo_root()
    for p in protocols:
        if not isinstance(p, dict):
            continue
        if not bool(p.get("enabled", True)):
            continue
        proto_name = str(p.get("name", "")).strip()
        if not proto_name:
            continue
        split = str(p.get("split", "val")).strip()
        ablation_dir = Path(str(p.get("ablation_dir", ""))).expanduser()
        if not ablation_dir.is_absolute():
            ablation_dir = (mia_root / ablation_dir).resolve()
        train_rel = str(p.get("train_filelist", "train.txt"))
        train_filelist = ablation_dir / train_rel
        val_filelists_cfg = p.get("val_filelists", None)
        if isinstance(val_filelists_cfg, (list, tuple)) and val_filelists_cfg:
            val_filelists = [ablation_dir / str(x) for x in val_filelists_cfg]
        else:
            glob_pat = str(p.get("val_glob", "val*.txt"))
            val_filelists = sorted([Path(x) for x in ablation_dir.glob(glob_pat) if Path(x).is_file()])
        for vf in val_filelists:
            cases.append(EvalCase(protocol=proto_name, train_filelist=train_filelist, val_filelist=vf, split=split))
    return cases


def _build_loader(
    *,
    filelist_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    step: int,
    std: bool,
    cond: bool,
    percent: float,
    max_samples: Optional[int],
    device: torch.device,
) -> DataLoader:
    from musclesinaction.dataloader.data import MyMuscleDataset, _seed_worker  # type: ignore

    dset = MyMuscleDataset(
        str(filelist_path),
        _NullLogger(),
        str(split),
        percent=float(percent),
        step=int(step),
        std="True" if bool(std) else "False",
        cond="True" if bool(cond) else "False",
        transform=None,
    )
    if max_samples is not None:
        max_n = min(len(dset), int(max_samples))
        dset = Subset(dset, range(max_n))
    return DataLoader(
        dset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        worker_init_fn=_seed_worker,
        pin_memory=(device.type == "cuda"),
    )


def _eval_stage2_on_filelist(
    *,
    model: Stage2Pose2EMG,
    train_cfg: Dict[str, Any],
    stage1: FrameCodebookModel,
    loader: DataLoader,
    device: torch.device,
    joints3d_root_center: bool,
    joints3d_root_index: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    pred_mode = str(train_cfg.get("model", {}).get("emg_pred_mode", "full")).strip().lower()
    emg_normalize_target = bool(train_cfg.get("data", {}).get("emg_normalize_target", False))
    stage1_emg = getattr(stage1, "emg", None)
    emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
    use_emg_denorm = (pred_mode == "full" and emg_normalize_target and emg_standardizer is not None)

    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    sample_counter = 0

    for batch in tqdm(loader, desc="stage2", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        
        cond = torch.as_tensor(batch.get("condval", None)) if batch.get("condval", None) is not None else None
        if cond is not None:
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.ndim == 1:
                cond = cond.unsqueeze(1)

        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()
        with torch.no_grad():
            out = model(joints3d, cond=cond)
            pred_bt8 = out["emg_pred"]
            if use_emg_denorm:
                mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(pred_bt8.shape[1]), device=device)
                pred_bt8 = pred_bt8 * std_bt8 + mean_bt8
        pred_np = pred_bt8.detach().cpu().numpy()
        gt_np = gt_bt8.detach().cpu().numpy()
        pred_all.append(pred_np.reshape(pred_np.shape[0] * pred_np.shape[1], 8))
        gt_all.append(gt_np.reshape(gt_np.shape[0] * gt_np.shape[1], 8))

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

    pred_cat = np.concatenate(pred_all, axis=0)
    gt_cat = np.concatenate(gt_all, axis=0)
    return pred_cat, gt_cat, per_sequence_rows


def _eval_official_model_on_filelist(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    sample_counter = 0

    for batch in tqdm(loader, desc="official", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        cond = torch.as_tensor(batch["condval"], device=device, dtype=torch.float32)
        if cond.ndim == 1:
            cond = cond.unsqueeze(1)

        b, t, j, c = joints3d.shape
        if (j, c) != (25, 3):
            raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
        if emg.shape != (b, 8, t):
            raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")

        skeleton = joints3d.reshape(b, t, 75)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()

        with torch.no_grad():
            pred_b8t = model(skeleton, cond)
        if pred_b8t.ndim != 3 or pred_b8t.shape != (b, 8, t):
            raise RuntimeError(f"Unexpected official model output shape: {tuple(pred_b8t.shape)}")
        pred_bt8 = pred_b8t.permute(0, 2, 1).contiguous()

        pred_np = pred_bt8.detach().cpu().numpy()
        gt_np = gt_bt8.detach().cpu().numpy()
        pred_all.append(pred_np.reshape(b * t, 8))
        gt_all.append(gt_np.reshape(b * t, 8))

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

    pred_cat = np.concatenate(pred_all, axis=0)
    gt_cat = np.concatenate(gt_all, axis=0)
    return pred_cat, gt_cat, per_sequence_rows


def _build_retrieval_db(
    *,
    loader: DataLoader,
    device: torch.device,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    seen = 0
    for batch in tqdm(loader, desc="retrieval_db", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()
        b = int(joints3d.shape[0])
        x = joints3d.reshape(b, -1).contiguous()
        y = gt_bt8.reshape(b, -1).contiguous()
        xs.append(x)
        ys.append(y)
        seen += b
        if max_samples is not None and seen >= int(max_samples):
            break
    xtr = torch.cat(xs, dim=0)
    ytr = torch.cat(ys, dim=0)
    if max_samples is not None and int(xtr.shape[0]) > int(max_samples):
        xtr = xtr[: int(max_samples)]
        ytr = ytr[: int(max_samples)]
    return xtr, ytr


def _predict_retrieval(
    *,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xq: torch.Tensor,
    train_chunk_size: int,
) -> torch.Tensor:
    if xq.ndim != 2 or xtr.ndim != 2:
        raise ValueError(f"Expected xq/xtr 2D, got {tuple(xq.shape)} {tuple(xtr.shape)}")
    q_norm = (xq * xq).sum(dim=1, keepdim=True)
    best_dist = torch.full((xq.shape[0],), float("inf"), device=xq.device, dtype=torch.float32)
    best_idx = torch.zeros((xq.shape[0],), device=xq.device, dtype=torch.long)
    n_train = int(xtr.shape[0])
    for start in range(0, n_train, int(train_chunk_size)):
        end = min(start + int(train_chunk_size), n_train)
        chunk = xtr[start:end]
        chunk_norm = (chunk * chunk).sum(dim=1).unsqueeze(0)
        prod = xq @ chunk.t()
        dist2 = q_norm + chunk_norm - 2.0 * prod
        idx_in_chunk = torch.argmin(dist2, dim=1)
        dist_in_chunk = dist2.gather(1, idx_in_chunk.view(-1, 1)).squeeze(1)
        better = dist_in_chunk < best_dist
        if better.any():
            best_dist = torch.where(better, dist_in_chunk, best_dist)
            best_idx = torch.where(better, idx_in_chunk + start, best_idx)
    return ytr[best_idx]


def _eval_retrieval_on_filelist(
    *,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    train_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    sample_counter = 0

    for batch in tqdm(loader, desc="retrieval", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()
        b, t, _ = gt_bt8.shape
        xq = joints3d.reshape(b, -1).contiguous()
        with torch.no_grad():
            ypred_flat = _predict_retrieval(xtr=xtr, ytr=ytr, xq=xq, train_chunk_size=train_chunk_size)
        pred_bt8 = ypred_flat.view(b, t, 8)

        pred_np = pred_bt8.detach().cpu().numpy()
        gt_np = gt_bt8.detach().cpu().numpy()
        pred_all.append(pred_np.reshape(b * t, 8))
        gt_all.append(gt_np.reshape(b * t, 8))

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

    pred_cat = np.concatenate(pred_all, axis=0)
    gt_cat = np.concatenate(gt_all, axis=0)
    return pred_cat, gt_cat, per_sequence_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="custom/configs/eval/eval.yaml",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    runtime = cfg.get("runtime", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    methods_cfg = cfg.get("methods", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    device = torch.device("cuda" if (str(runtime.get("device", "cuda")) == "cuda" and torch.cuda.is_available()) else "cpu")
    num_workers = int(runtime.get("num_workers", 4))
    batch_size = int(runtime.get("batch_size", 8))
    step = int(data_cfg.get("step", 30))
    std = bool(data_cfg.get("std", False))
    percent = float(data_cfg.get("percent", 1.0))
    max_samples = data_cfg.get("max_samples", None)
    max_samples = int(max_samples) if max_samples is not None else None
    joints3d_root_center = bool(data_cfg.get("joints3d_root_center", True))
    joints3d_root_index = int(data_cfg.get("joints3d_root_index", 8))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_cfg.get("out_dir", None)
    if out_dir:
        out_root = Path(str(out_dir)).expanduser()
        out_root = out_root if out_root.is_absolute() else (mia_root / out_root).resolve()
    else:
        out_root = (mia_root / "custom" / "output" / "mia_style_eval" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cases = _discover_eval_cases(cfg)
    if not cases:
        raise RuntimeError("No eval cases found. Configure protocols in eval.yaml.")

    stage2_enabled = bool((methods_cfg.get("stage2", {}) or {}).get("enabled", True))
    official_cond_enabled = bool((methods_cfg.get("official_cond", {}) or {}).get("enabled", True))
    official_nocond_enabled = bool((methods_cfg.get("official_nocond", {}) or {}).get("enabled", True))
    retrieval_enabled = bool((methods_cfg.get("retrieval", {}) or {}).get("enabled", True))

    stage2_model = None
    stage2_train_cfg = None
    stage2_stage1 = None
    stage2_ckpt_path = None
    if stage2_enabled:
        m = methods_cfg.get("stage2", {}) or {}
        ckpt = Path(str(m.get("checkpoint", ""))).expanduser()
        stage2_ckpt_path = ckpt if ckpt.is_absolute() else (mia_root / ckpt).resolve()
        if not stage2_ckpt_path.exists():
            raise FileNotFoundError(f"Stage2 checkpoint not found: {stage2_ckpt_path}")
        stage1_override = m.get("stage1_checkpoint", None)
        stage1_override_path = None
        if stage1_override:
            p = Path(str(stage1_override)).expanduser()
            stage1_override_path = p if p.is_absolute() else (mia_root / p).resolve()
        stage2_model, stage2_train_cfg, _, stage2_stage1 = _build_stage2_from_ckpt(
            stage2_ckpt_path, device=device, stage1_override_ckpt=stage1_override_path
        )

    official_cond_model = None
    official_cond_ckpt = None
    if official_cond_enabled:
        m = methods_cfg.get("official_cond", {}) or {}
        ckpt = Path(str(m.get("checkpoint", "pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth"))).expanduser()
        official_cond_ckpt = ckpt if ckpt.is_absolute() else (mia_root / ckpt).resolve()
        if not official_cond_ckpt.exists():
            raise FileNotFoundError(f"Official cond checkpoint not found: {official_cond_ckpt}")
        official_cond_model, _, _ = _load_official_model(official_cond_ckpt, device=device)

    official_nocond_model = None
    official_nocond_ckpt = None
    if official_nocond_enabled:
        m = methods_cfg.get("official_nocond", {}) or {}
        cfg_path_val = str(m.get("checkpoint", ""))
        
        # 如果配置文件里没有配，或者配了但是文件不存在（比如之前默认的 pretrained-checkpoints 里的 nocond 路径）
        # 则使用我们自己训练的 nocond 权重路径作为 fallback。
        fallback_path = "/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/output/20260410_102342/checkpoints/generalization_new_nocond_clean_posetoemg/model_100.pth"
        
        if cfg_path_val:
            ckpt = Path(cfg_path_val).expanduser()
            official_nocond_ckpt = ckpt if ckpt.is_absolute() else (mia_root / ckpt).resolve()
            
        if not official_nocond_ckpt or not official_nocond_ckpt.exists():
            official_nocond_ckpt = Path(fallback_path).resolve()
            
        if not official_nocond_ckpt.exists():
            raise FileNotFoundError(f"Official nocond checkpoint not found: {official_nocond_ckpt}")
        official_nocond_model, _, _ = _load_official_model(official_nocond_ckpt, device=device)

    retrieval_cfg = methods_cfg.get("retrieval", {}) or {}
    retrieval_device = torch.device(str((runtime.get("retrieval", {}) or {}).get("device", "cpu")))
    retrieval_query_bs = int((runtime.get("retrieval", {}) or {}).get("query_batch_size", batch_size))
    retrieval_train_chunk = int((runtime.get("retrieval", {}) or {}).get("train_chunk_size", 2048))
    retrieval_max_train = retrieval_cfg.get("max_train_samples", None)
    retrieval_max_train = int(retrieval_max_train) if retrieval_max_train is not None else None

    per_case_summary_rows: List[Dict[str, Any]] = []
    summary_fieldnames = [
        "protocol",
        "val_filelist",
        "method",
        "official_global_rmse",
        "official_global_mse",
        "official_global_mae",
        "num_sequences",
        "num_frames",
        "out_dir",
    ]

    retrieval_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for case in cases:
        val_name = case.val_filelist.stem
        case_dir = out_root / case.protocol / val_name
        case_dir.mkdir(parents=True, exist_ok=True)

        if stage2_enabled:
            loader = _build_loader(
                filelist_path=case.val_filelist,
                split=case.split,
                batch_size=batch_size,
                num_workers=num_workers,
                step=step,
                std=std,
                cond=True,
                percent=percent,
                max_samples=max_samples,
                device=device,
            )
            pred_cat, gt_cat, per_seq = _eval_stage2_on_filelist(
                model=stage2_model,
                train_cfg=stage2_train_cfg,
                stage1=stage2_stage1,
                loader=loader,
                device=device,
                joints3d_root_center=joints3d_root_center,
                joints3d_root_index=joints3d_root_index,
            )
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "stage2"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                [
                    "eval_index",
                    "filepath",
                    "frames",
                    "mse",
                    "rmse",
                    "mae",
                    *[f"rmse_{n}" for n in MUSCLE_NAMES],
                ],
            )
            summary = {
                "config_path": str(cfg_path),
                "protocol": case.protocol,
                "split": case.split,
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "method": "stage2",
                "stage2_checkpoint": str(stage2_ckpt_path) if stage2_ckpt_path else None,
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            per_case_summary_rows.append(
                {
                    "protocol": case.protocol,
                    "val_filelist": val_name,
                    "method": "stage2",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                }
            )

        if official_cond_enabled:
            loader = _build_loader(
                filelist_path=case.val_filelist,
                split=case.split,
                batch_size=batch_size,
                num_workers=num_workers,
                step=step,
                std=std,
                cond=True,
                percent=percent,
                max_samples=max_samples,
                device=device,
            )
            pred_cat, gt_cat, per_seq = _eval_official_model_on_filelist(model=official_cond_model, loader=loader, device=device)
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "official_cond"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                [
                    "eval_index",
                    "filepath",
                    "frames",
                    "mse",
                    "rmse",
                    "mae",
                    *[f"rmse_{n}" for n in MUSCLE_NAMES],
                ],
            )
            summary = {
                "config_path": str(cfg_path),
                "protocol": case.protocol,
                "split": case.split,
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "method": "official_cond",
                "checkpoint": str(official_cond_ckpt) if official_cond_ckpt else None,
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            per_case_summary_rows.append(
                {
                    "protocol": case.protocol,
                    "val_filelist": val_name,
                    "method": "official_cond",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                }
            )

        if official_nocond_enabled:
            loader = _build_loader(
                filelist_path=case.val_filelist,
                split=case.split,
                batch_size=batch_size,
                num_workers=num_workers,
                step=step,
                std=std,
                cond=False,
                percent=percent,
                max_samples=max_samples,
                device=device,
            )
            pred_cat, gt_cat, per_seq = _eval_official_model_on_filelist(model=official_nocond_model, loader=loader, device=device)
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "official_nocond"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                [
                    "eval_index",
                    "filepath",
                    "frames",
                    "mse",
                    "rmse",
                    "mae",
                    *[f"rmse_{n}" for n in MUSCLE_NAMES],
                ],
            )
            summary = {
                "config_path": str(cfg_path),
                "protocol": case.protocol,
                "split": case.split,
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "method": "official_nocond",
                "checkpoint": str(official_nocond_ckpt) if official_nocond_ckpt else None,
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            per_case_summary_rows.append(
                {
                    "protocol": case.protocol,
                    "val_filelist": val_name,
                    "method": "official_nocond",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                }
            )

        if retrieval_enabled:
            cache_key = f"{case.protocol}::{str(case.train_filelist)}"
            if cache_key not in retrieval_cache:
                train_loader = _build_loader(
                    filelist_path=case.train_filelist,
                    split="train",
                    batch_size=int(batch_size),
                    num_workers=num_workers,
                    step=step,
                    std=std,
                    cond=False,
                    percent=percent,
                    max_samples=retrieval_max_train,
                    device=retrieval_device,
                )
                xtr, ytr = _build_retrieval_db(
                    loader=train_loader,
                    device=retrieval_device,
                    joints3d_root_center=joints3d_root_center,
                    joints3d_root_index=joints3d_root_index,
                    max_samples=retrieval_max_train,
                )
                retrieval_cache[cache_key] = (xtr, ytr)
            xtr, ytr = retrieval_cache[cache_key]
            val_loader = _build_loader(
                filelist_path=case.val_filelist,
                split=case.split,
                batch_size=int(retrieval_query_bs),
                num_workers=num_workers,
                step=step,
                std=std,
                cond=False,
                percent=percent,
                max_samples=max_samples,
                device=retrieval_device,
            )
            pred_cat, gt_cat, per_seq = _eval_retrieval_on_filelist(
                xtr=xtr,
                ytr=ytr,
                loader=val_loader,
                device=retrieval_device,
                joints3d_root_center=joints3d_root_center,
                joints3d_root_index=joints3d_root_index,
                train_chunk_size=retrieval_train_chunk,
            )
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "retrieval"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                [
                    "eval_index",
                    "filepath",
                    "frames",
                    "mse",
                    "rmse",
                    "mae",
                    *[f"rmse_{n}" for n in MUSCLE_NAMES],
                ],
            )
            summary = {
                "config_path": str(cfg_path),
                "protocol": case.protocol,
                "split": case.split,
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "method": "retrieval",
                "max_train_samples": retrieval_max_train,
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            per_case_summary_rows.append(
                {
                    "protocol": case.protocol,
                    "val_filelist": val_name,
                    "method": "retrieval",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                }
            )

    summary_csv = out_root / "summary.csv"
    _write_csv_rows(summary_csv, per_case_summary_rows, summary_fieldnames)
    print("[MiaStyleEval] out_root =", out_root)
    print("[MiaStyleEval] summary_csv =", summary_csv)


if __name__ == "__main__":
    main()
