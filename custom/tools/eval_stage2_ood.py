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
from typing import Any, Dict, List, Optional, Sequence, Tuple
    
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


def _root_center_joints3d(joints3d: torch.Tensor, root_index: int) -> torch.Tensor:
    root = joints3d[:, :, root_index : root_index + 1, :]
    return joints3d - root


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
            raise ValueError("Missing stage1 checkpoint in stage2 checkpoint payload.")
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
    model = TransformerEnc(**model_args).to(device)
    state = payload.get("my_model", None)
    if state is None:
        state = {k: v for k, v in payload.items() if isinstance(k, str)}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@dataclass(frozen=True)
class OODCase:
    protocol: str
    target: str
    train_filelist: Path
    val_filelist: Path


def _discover_targets(protocol_cfg: Dict[str, Any]) -> List[str]:
    if protocol_cfg.get("targets", None):
        return [str(x).strip() for x in list(protocol_cfg["targets"]) if str(x).strip()]
    ablation_dir = Path(str(protocol_cfg["ablation_dir"])).expanduser().resolve()
    glob_pat = str(protocol_cfg.get("val_glob", "val_*.txt"))
    files = sorted([p for p in ablation_dir.glob(glob_pat) if p.is_file()])
    targets: List[str] = []
    for p in files:
        stem = p.stem
        if stem.startswith("val_"):
            t = stem[len("val_") :]
        elif stem.startswith("val"):
            t = stem[len("val") :]
        else:
            continue
        t = t.strip("_").strip()
        if t:
            targets.append(t)
    return targets


def _resolve_case_paths(protocol_cfg: Dict[str, Any], target: str) -> tuple[Path, Path]:
    ablation_dir = Path(str(protocol_cfg["ablation_dir"])).expanduser().resolve()
    train_pat = str(protocol_cfg.get("train_pattern", "train_{target}.txt"))
    val_pat = str(protocol_cfg.get("val_pattern", "val_{target}.txt"))
    train_file = (ablation_dir / train_pat.format(target=target)).resolve()
    val_file = (ablation_dir / val_pat.format(target=target)).resolve()
    return train_file, val_file


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


def _eval_stage2_on_val(
    *,
    model: Stage2Pose2EMG,
    train_cfg: Dict[str, Any],
    stage1: FrameCodebookModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    pred_mode = str(train_cfg.get("model", {}).get("emg_pred_mode", "full")).strip().lower()
    emg_normalize_target = bool(train_cfg.get("data", {}).get("emg_normalize_target", False))
    root_center = bool(train_cfg.get("data", {}).get("joints3d_root_center", False))
    root_index = int(train_cfg.get("data", {}).get("joints3d_root_index", 8))

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
        if root_center:
            joints3d = _root_center_joints3d(joints3d, root_index)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()
        with torch.no_grad():
            out = model(joints3d)
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


def _eval_official_on_val(
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
        if emg.shape != (b, 8, t):
            raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")
        skeleton = joints3d.reshape(b, t, 75)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()
        with torch.no_grad():
            pred_b8t = model(skeleton, cond)
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
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    seen = 0
    for batch in tqdm(loader, desc="retrieval_db", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        gt_bt8 = emg.permute(0, 2, 1).contiguous()
        b = int(joints3d.shape[0])
        xs.append(joints3d.reshape(b, -1).contiguous())
        ys.append(gt_bt8.reshape(b, -1).contiguous())
        seen += b
        if max_samples is not None and seen >= int(max_samples):
            break
    xtr = torch.cat(xs, dim=0)
    ytr = torch.cat(ys, dim=0)
    if max_samples is not None and int(xtr.shape[0]) > int(max_samples):
        xtr = xtr[: int(max_samples)]
        ytr = ytr[: int(max_samples)]
    return xtr, ytr


def _predict_retrieval(xtr: torch.Tensor, ytr: torch.Tensor, xq: torch.Tensor, train_chunk_size: int) -> torch.Tensor:
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


def _eval_retrieval_on_val(
    *,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    train_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    sample_counter = 0

    for batch in tqdm(loader, desc="retrieval", unit="batch"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
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
    parser.add_argument("--config", type=str, default="custom/configs/ood/ood_eval.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    ood_train_cfg_path = Path(str(cfg.get("ood_train_config", ""))).expanduser()
    ood_train_cfg_path = ood_train_cfg_path if ood_train_cfg_path.is_absolute() else (mia_root / ood_train_cfg_path).resolve()
    if not ood_train_cfg_path.exists():
        raise FileNotFoundError(f"ood_train_config not found: {ood_train_cfg_path}")
    ood_train_cfg = _load_yaml(ood_train_cfg_path)

    ckpt_root = Path(str(ood_train_cfg.get("output", {}).get("ckpt_root", ""))).expanduser()
    ckpt_root = ckpt_root if ckpt_root.is_absolute() else (mia_root / ckpt_root).resolve()

    runtime = cfg.get("runtime", {}) or {}
    device = torch.device("cuda" if (str(runtime.get("device", "cuda")) == "cuda" and torch.cuda.is_available()) else "cpu")
    num_workers = int(runtime.get("num_workers", 4))
    batch_size = int(runtime.get("batch_size", 8))
    step = int(runtime.get("step", 30))
    std = bool(runtime.get("std", False))
    percent = float(runtime.get("percent", 1.0))
    max_samples = runtime.get("max_samples", None)
    max_samples = int(max_samples) if max_samples is not None else None

    filters = cfg.get("filters", {}) or {}
    only_protocol = filters.get("protocol", None)
    only_target = filters.get("target", None)
    only_protocol = str(only_protocol).strip() if only_protocol not in (None, "", "null") else None
    only_target = str(only_target).strip() if only_target not in (None, "", "null") else None

    methods = cfg.get("methods", {}) or {}
    stage2_cfg = methods.get("stage2", {}) or {}
    stage2_enabled = bool(stage2_cfg.get("enabled", True))
    stage2_ckpt_name = str(stage2_cfg.get("ckpt_name", "best.pt")).strip()

    official_cfg = methods.get("official", {}) or {}
    official_enabled = bool(official_cfg.get("enabled", True))
    official_eval_cond = bool(official_cfg.get("eval_cond", True))
    official_eval_nocond = bool(official_cfg.get("eval_nocond", True))
    official_ckpt_str = str(official_cfg.get("checkpoint", "pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth")).strip()
    official_ckpt = Path(official_ckpt_str).expanduser()
    official_ckpt = official_ckpt if official_ckpt.is_absolute() else (mia_root / official_ckpt).resolve()
    if official_enabled and not official_ckpt.exists():
        raise FileNotFoundError(f"Official checkpoint not found: {official_ckpt}")

    retrieval_cfg = methods.get("retrieval", {}) or {}
    retrieval_enabled = bool(retrieval_cfg.get("enabled", True))
    retrieval_rt = runtime.get("retrieval", {}) or {}
    retrieval_device = torch.device(str(retrieval_rt.get("device", "cpu")))
    retrieval_query_bs = int(retrieval_rt.get("query_batch_size", batch_size))
    retrieval_train_chunk = int(retrieval_rt.get("train_chunk_size", 2048))
    retrieval_max_train = retrieval_rt.get("max_train_samples", None)
    retrieval_max_train = int(retrieval_max_train) if retrieval_max_train is not None else None

    out_cfg = cfg.get("output", {}) or {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_cfg.get("out_dir", None)
    if out_dir:
        out_root = Path(str(out_dir)).expanduser()
        out_root = out_root if out_root.is_absolute() else (mia_root / out_root).resolve()
    else:
        out_root = (mia_root / "custom" / "output" / "ood_eval" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    official_model = _load_official_model(official_ckpt, device=device) if official_enabled else None

    cases: List[OODCase] = []
    for p in ood_train_cfg.get("protocols", []) or []:
        if not isinstance(p, dict) or not bool(p.get("enabled", True)):
            continue
        proto = str(p.get("name", "")).strip()
        if not proto:
            continue
        if only_protocol is not None and proto != only_protocol:
            continue
        targets = _discover_targets(p)
        if only_target is not None:
            targets = [t for t in targets if t == only_target]
        for t in targets:
            train_file, val_file = _resolve_case_paths(p, t)
            cases.append(OODCase(protocol=proto, target=t, train_filelist=train_file, val_filelist=val_file))

    if not cases:
        raise RuntimeError("No OOD cases to evaluate (check filters/protocols).")

    summary_rows: List[Dict[str, Any]] = []
    summary_fields = [
        "protocol",
        "target",
        "method",
        "official_global_rmse",
        "official_global_mse",
        "official_global_mae",
        "num_sequences",
        "num_frames",
        "out_dir",
        "val_filelist",
        "train_filelist",
        "stage2_checkpoint",
        "official_checkpoint",
    ]

    retrieval_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for case in cases:
        case_dir = out_root / case.protocol / case.target
        case_dir.mkdir(parents=True, exist_ok=True)

        if stage2_enabled:
            stage2_dir = (ckpt_root / case.protocol / case.target).resolve()
            stage2_ckpt = stage2_dir / stage2_ckpt_name
            if stage2_ckpt.exists():
                stage2_model, stage2_train_cfg, _, stage2_stage1 = _build_stage2_from_ckpt(stage2_ckpt, device=device, stage1_override_ckpt=None)
                loader = _build_loader(
                    filelist_path=case.val_filelist,
                    split="val",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    step=step,
                    std=std,
                    cond=True,
                    percent=percent,
                    max_samples=max_samples,
                    device=device,
                )
                pred_cat, gt_cat, per_seq = _eval_stage2_on_val(
                    model=stage2_model,
                    train_cfg=stage2_train_cfg,
                    stage1=stage2_stage1,
                    loader=loader,
                    device=device,
                )
                metrics = _compute_official_metrics(pred_cat, gt_cat)
                run_dir = case_dir / "stage2"
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_csv_rows(
                    run_dir / "per_sequence.csv",
                    per_seq,
                    ["eval_index", "filepath", "frames", "mse", "rmse", "mae", *[f"rmse_{n}" for n in MUSCLE_NAMES]],
                )
                summary = {
                    "config_path": str(cfg_path),
                    "ood_train_config": str(ood_train_cfg_path),
                    "protocol": case.protocol,
                    "target": case.target,
                    "method": "stage2",
                    "train_filelist": str(case.train_filelist),
                    "val_filelist": str(case.val_filelist),
                    "stage2_checkpoint": str(stage2_ckpt),
                    **metrics,
                    "num_sequences": int(len(per_seq)),
                }
                with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                summary_rows.append(
                    {
                        "protocol": case.protocol,
                        "target": case.target,
                        "method": "stage2",
                        "official_global_rmse": summary["official_global_rmse"],
                        "official_global_mse": summary["official_global_mse"],
                        "official_global_mae": summary["official_global_mae"],
                        "num_sequences": summary["num_sequences"],
                        "num_frames": summary["num_frames"],
                        "out_dir": str(run_dir),
                        "val_filelist": str(case.val_filelist),
                        "train_filelist": str(case.train_filelist),
                        "stage2_checkpoint": str(stage2_ckpt),
                        "official_checkpoint": "",
                    }
                )

        if official_enabled and official_eval_cond:
            loader = _build_loader(
                filelist_path=case.val_filelist,
                split="val",
                batch_size=batch_size,
                num_workers=num_workers,
                step=step,
                std=std,
                cond=True,
                percent=percent,
                max_samples=max_samples,
                device=device,
            )
            pred_cat, gt_cat, per_seq = _eval_official_on_val(model=official_model, loader=loader, device=device)
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "official_cond"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                ["eval_index", "filepath", "frames", "mse", "rmse", "mae", *[f"rmse_{n}" for n in MUSCLE_NAMES]],
            )
            summary = {
                "config_path": str(cfg_path),
                "ood_train_config": str(ood_train_cfg_path),
                "protocol": case.protocol,
                "target": case.target,
                "method": "official_cond",
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "official_checkpoint": str(official_ckpt),
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary_rows.append(
                {
                    "protocol": case.protocol,
                    "target": case.target,
                    "method": "official_cond",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                    "val_filelist": str(case.val_filelist),
                    "train_filelist": str(case.train_filelist),
                    "stage2_checkpoint": "",
                    "official_checkpoint": str(official_ckpt),
                }
            )

        if official_enabled and official_eval_nocond:
            loader = _build_loader(
                filelist_path=case.val_filelist,
                split="val",
                batch_size=batch_size,
                num_workers=num_workers,
                step=step,
                std=std,
                cond=False,
                percent=percent,
                max_samples=max_samples,
                device=device,
            )
            pred_cat, gt_cat, per_seq = _eval_official_on_val(model=official_model, loader=loader, device=device)
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "official_nocond"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                ["eval_index", "filepath", "frames", "mse", "rmse", "mae", *[f"rmse_{n}" for n in MUSCLE_NAMES]],
            )
            summary = {
                "config_path": str(cfg_path),
                "ood_train_config": str(ood_train_cfg_path),
                "protocol": case.protocol,
                "target": case.target,
                "method": "official_nocond",
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "official_checkpoint": str(official_ckpt),
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary_rows.append(
                {
                    "protocol": case.protocol,
                    "target": case.target,
                    "method": "official_nocond",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                    "val_filelist": str(case.val_filelist),
                    "train_filelist": str(case.train_filelist),
                    "stage2_checkpoint": "",
                    "official_checkpoint": str(official_ckpt),
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
                xtr, ytr = _build_retrieval_db(loader=train_loader, device=retrieval_device, max_samples=retrieval_max_train)
                retrieval_cache[cache_key] = (xtr, ytr)
            xtr, ytr = retrieval_cache[cache_key]
            val_loader = _build_loader(
                filelist_path=case.val_filelist,
                split="val",
                batch_size=int(retrieval_query_bs),
                num_workers=num_workers,
                step=step,
                std=std,
                cond=False,
                percent=percent,
                max_samples=max_samples,
                device=retrieval_device,
            )
            pred_cat, gt_cat, per_seq = _eval_retrieval_on_val(
                xtr=xtr, ytr=ytr, loader=val_loader, device=retrieval_device, train_chunk_size=retrieval_train_chunk
            )
            metrics = _compute_official_metrics(pred_cat, gt_cat)
            run_dir = case_dir / "retrieval"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv_rows(
                run_dir / "per_sequence.csv",
                per_seq,
                ["eval_index", "filepath", "frames", "mse", "rmse", "mae", *[f"rmse_{n}" for n in MUSCLE_NAMES]],
            )
            summary = {
                "config_path": str(cfg_path),
                "ood_train_config": str(ood_train_cfg_path),
                "protocol": case.protocol,
                "target": case.target,
                "method": "retrieval",
                "train_filelist": str(case.train_filelist),
                "val_filelist": str(case.val_filelist),
                "max_train_samples": retrieval_max_train,
                **metrics,
                "num_sequences": int(len(per_seq)),
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary_rows.append(
                {
                    "protocol": case.protocol,
                    "target": case.target,
                    "method": "retrieval",
                    "official_global_rmse": summary["official_global_rmse"],
                    "official_global_mse": summary["official_global_mse"],
                    "official_global_mae": summary["official_global_mae"],
                    "num_sequences": summary["num_sequences"],
                    "num_frames": summary["num_frames"],
                    "out_dir": str(run_dir),
                    "val_filelist": str(case.val_filelist),
                    "train_filelist": str(case.train_filelist),
                    "stage2_checkpoint": "",
                    "official_checkpoint": "",
                }
            )

    summary_csv = out_root / "summary.csv"
    _write_csv_rows(summary_csv, summary_rows, summary_fields)
    print("[OOD-Eval] out_root =", out_root)
    print("[OOD-Eval] summary_csv =", summary_csv)


if __name__ == "__main__":
    main()
