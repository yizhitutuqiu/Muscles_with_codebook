from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# File path:
#   .../golf_third_party/musclesinaction/custom/tools/eval_frame_codebook_official_metrics.py
_MIA_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.models.frame_codebook import FrameCodebookConfig, FrameCodebookModel, ModalityConfig  # noqa: E402
from custom.models.temporal import TemporalConfig  # noqa: E402
from custom.models.vq_ema import VQEMAConfig  # noqa: E402
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


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _apply_runtime_overrides(cfg: Dict[str, Any], *, emg_online_std: bool) -> Dict[str, Any]:
    model_cfg = cfg.setdefault("model", {})
    modalities = model_cfg.setdefault("modalities", {})
    emg_cfg = modalities.setdefault("emg", {})
    emg_cfg["online_std"] = bool(emg_online_std)
    return cfg


def _build_modality_cfg(name: str, d: Dict[str, Any]) -> ModalityConfig:
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


def _build_optional_modality_cfg(name: str, d: Dict[str, Any] | None) -> Optional[ModalityConfig]:
    if not d:
        return None
    if not bool(d.get("enabled", True)):
        return None
    return _build_modality_cfg(name, d)


def _build_model_from_cfg(cfg: Dict[str, Any], device: torch.device) -> FrameCodebookModel:
    model_cfg = cfg["model"]
    vq_kwargs = dict(model_cfg["vq"])
    if "beta" not in vq_kwargs and "commitment_weight" in vq_kwargs:
        vq_kwargs["beta"] = vq_kwargs.pop("commitment_weight")
    vq_cfg = VQEMAConfig(**vq_kwargs)
    mods = model_cfg["modalities"]
    fc_cfg = FrameCodebookConfig(
        vq=vq_cfg,
        joints3d=_build_modality_cfg("joints3d", mods["joints3d"]),
        smpl_pose=_build_optional_modality_cfg("smpl_pose", mods.get("smpl_pose")),
        emg=_build_modality_cfg("emg", mods["emg"]),
        encoder_decoder_only=bool(model_cfg.get("encoder_decoder_only", False)),
    )
    return FrameCodebookModel(fc_cfg).to(device)


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


def _list_ckpt_dir(ckpt_dir: Path) -> List[Path]:
    """List .pt files in experiment root and in ckpt/ subdir (step_*.pt)."""
    out: List[Path] = []
    for p in ckpt_dir.glob("*.pt"):
        if p.is_file():
            out.append(p.resolve())
    step_sub = ckpt_dir / "ckpt"
    if step_sub.is_dir():
        for p in step_sub.glob("*.pt"):
            if p.is_file():
                out.append(p.resolve())
    return out


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = _list_ckpt_dir(ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir} or {ckpt_dir / 'ckpt'}")
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def _collect_checkpoints(*, checkpoint: str | None, checkpoint_dir: str | None, all_checkpoints: bool, mia_root: Path) -> List[Path]:
    if checkpoint:
        return [Path(checkpoint).expanduser().resolve()]

    if checkpoint_dir:
        ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    else:
        ckpt_dir = mia_root / "custom" / "checkpoints" / "frame_codebook"

    ckpts = _list_ckpt_dir(ckpt_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir} or {ckpt_dir / 'ckpt'}")
    ckpts.sort(key=lambda p: p.name)
    if all_checkpoints:
        return ckpts
    return [_find_latest_checkpoint(ckpt_dir)]


def _maybe_build_filelist(
    *,
    mia_root: Path,
    split: str,
    out_txt: Path,
    max_samples: int | None,
    require_files: Sequence[str],
) -> Path:
    # 当指定 max_samples 时用独立路径并强制重建，避免复用已有“全量”filelist 导致上限不生效
    if max_samples is not None:
        out_txt = out_txt.parent / f"{out_txt.stem}_n{max_samples}{out_txt.suffix}"
        res = build_mia_train_filelist(
            mia_repo_root=mia_root,
            split=split,
            out_txt=out_txt,
            max_samples=max_samples,
            require_files=list(require_files),
        )
        print(f"[FileList] wrote {res.num_samples} samples (max_samples={max_samples}) -> {res.filelist_path}")
        return res.filelist_path
    if out_txt.exists() and out_txt.stat().st_size > 0:
        return out_txt
    res = build_mia_train_filelist(
        mia_repo_root=mia_root,
        split=split,
        out_txt=out_txt,
        max_samples=max_samples,
        require_files=list(require_files),
    )
    print(f"[FileList] wrote {res.num_samples} samples -> {res.filelist_path}")
    return res.filelist_path


def _to_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def _root_center_joints3d(joints3d: torch.Tensor, root_index: int) -> torch.Tensor:
    if joints3d.ndim != 4 or joints3d.shape[-2:] != (25, 3):
        raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
    if not (0 <= int(root_index) < int(joints3d.shape[2])):
        raise ValueError(f"root_index out of range: {root_index}")
    root = joints3d[:, :, root_index : root_index + 1, :]
    return joints3d - root


def _prepare_eval_batch(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    pack_mode: str = "frame",
    clip_len: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    joints3d = _to_tensor(batch["3dskeleton"]).to(device=device, dtype=torch.float32)
    emg = _to_tensor(batch["emg_values"]).to(device=device, dtype=torch.float32)

    if joints3d.ndim != 4:
        raise ValueError(f"Expected 3dskeleton (B,T,25,3), got {tuple(joints3d.shape)}")
    if emg.ndim != 3:
        raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")

    b, t, j, c = joints3d.shape
    if (j, c) != (25, 3):
        raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
    if emg.shape != (b, 8, t):
        raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")

    if joints3d_root_center:
        joints3d = _root_center_joints3d(joints3d, joints3d_root_index)

    mode = str(pack_mode).lower().strip()
    if mode in ("frame", "frames"):
        joints3d_flat = joints3d.reshape(b * t, 75)
        emg_bt8 = emg.permute(0, 2, 1).contiguous()
        emg_flat = emg_bt8.reshape(b * t, 8)
        return joints3d_flat, emg_flat, b, t

    if mode in ("clip", "sequence", "seq"):
        # 与 train 一致：严格 30 帧一切片，取前 L 帧（无随机 crop、无重叠）；batch 内 B 个 clip 独立，TCN 无跨 batch 状态
        L = int(clip_len)
        if t < L:
            raise ValueError(f"Clip length mismatch: got T={t} < clip_len={L}")
        if t != L:
            joints3d = joints3d[:, :L]
            emg = emg[:, :, :L]
        joints3d_clip = joints3d.reshape(b, L, 75).contiguous().reshape(b, L * 75)
        emg_clip = emg.permute(0, 2, 1).contiguous().reshape(b, L * 8)
        return joints3d_clip, emg_clip, b, L

    raise ValueError(f"Unsupported data.pack.mode: {pack_mode}")


def _denormalize_from_modality(mod, x_n: torch.Tensor) -> torch.Tensor:
    stdzr = getattr(mod, "standardizer", None)
    if stdzr is None:
        return x_n
    std = torch.sqrt(torch.clamp_min(stdzr.var, stdzr.cfg.clip_std_min**2) + float(stdzr.cfg.eps))
    return x_n * std + stdzr.mean


@torch.no_grad()
def _predict_emg_from_joints_raw(
    model: FrameCodebookModel,
    joints3d_flat: torch.Tensor,
    *,
    bypass_vq: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xj_n, zq_j, idx_j, _, ppl_j, _ = model._encode_quantize(
        model.joints3d, joints3d_flat, bypass_vq=bypass_vq
    )
    del xj_n
    pred_emg_n = model._decode(model.emg, zq_j)
    pred_emg = _denormalize_from_modality(model.emg, pred_emg_n)
    return pred_emg, idx_j, ppl_j, zq_j


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


def _evaluate_one_checkpoint(
    *,
    ckpt_path: Path,
    fallback_config_path: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    out_root: Path,
    emg_online_std: bool,
    eval_in_train_mode: bool = False,
    stage2_ckpt_path: Path | None = None,
) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)}")

    if "config" in payload:
        cfg = payload["config"]
    else:
        cfg = _load_yaml(fallback_config_path)
    cfg = _apply_runtime_overrides(cfg, emg_online_std=bool(emg_online_std))

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    model = _build_model_from_cfg(cfg, device)
    state = payload["model_state"]
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        if "unexpected key" in str(e) or "missing key" in str(e):
            print("[Eval] load_state_dict(strict=True) 失败（可能为旧 checkpoint 含 TCN BatchNorm 的 running_*），改用 strict=False")
            load_result = model.load_state_dict(state, strict=False)
            # 兼容不同 PyTorch 版本：新版本返回 NamedTuple(missing_keys, unexpected_keys)，旧版可能返回 None
            if load_result is not None:
                missing_keys = list(getattr(load_result, "missing_keys", []))
                unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
            else:
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(state.keys())
                missing_keys = sorted(model_keys - ckpt_keys)
                unexpected_keys = sorted(ckpt_keys - model_keys)
            if missing_keys:
                print(f"!!! 警告: 缺失了 {len(missing_keys)} 个 Keys（未加载，可能为随机初始化）!!!")
                for k in missing_keys[:20]:
                    print(f"  missing: {k}")
                if len(missing_keys) > 20:
                    print(f"  ... 共 {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"[Eval] checkpoint 中多出 {len(unexpected_keys)} 个 Keys（已忽略）")
                for k in unexpected_keys[:10]:
                    print(f"  unexpected: {k}")
                if len(unexpected_keys) > 10:
                    print(f"  ... 共 {len(unexpected_keys)} 个")
        else:
            raise
    if eval_in_train_mode:
        model.train()
        print("[Eval] eval_in_train_mode=True: 使用 model.train() 跑评测（试金石：排查 OnlineStandardizer running stats）")
    else:
        model.eval()

    data_cfg = cfg.get("data", {})
    pack_cfg = (data_cfg.get("pack", {}) or {}) if isinstance(data_cfg, dict) else {}
    pack_mode = str(pack_cfg.get("mode", "frame")).strip().lower()
    clip_len = int(pack_cfg.get("clip_len", data_cfg.get("step", 30)))
    require_files = data_cfg.get("require_files", ["emgvalues.npy", "joints3d.npy"])

    # 与训练完全一致：split=train 且未限制 max_samples 时，使用 config 中的 train_filelist（与 train 同文件、同顺序）
    if split == "train" and max_samples is None:
        train_filelist = data_cfg.get("train_filelist")
        if train_filelist:
            filelist_path = (mia_root / train_filelist).resolve()
            if not filelist_path.exists() or filelist_path.stat().st_size == 0:
                res = build_mia_train_filelist(
                    mia_repo_root=mia_root,
                    split="train",
                    out_txt=filelist_path,
                    require_files=require_files,
                )
                print(f"[Eval] 使用与训练相同 filelist（train），重建 -> {filelist_path} ({res.num_samples} 条)")
            else:
                print(f"[Eval] 使用与训练相同 filelist（train）: {filelist_path}")
        else:
            filelist_path = mia_root / "custom" / "tools" / "datasetsplits" / f"miaofficial_{split}_eval.txt"
            filelist_path = _maybe_build_filelist(
                mia_root=mia_root,
                split=split,
                out_txt=filelist_path,
                max_samples=max_samples,
                require_files=require_files,
            )
    else:
        filelist_path = mia_root / "custom" / "tools" / "datasetsplits" / f"miaofficial_{split}_eval.txt"
        filelist_path = _maybe_build_filelist(
            mia_root=mia_root,
            split=split,
            out_txt=filelist_path,
            max_samples=max_samples,
            require_files=require_files,
        )

    # Train vs Eval 一致性（与 train_frame_codebook.py 对齐）：
    # - 数据：config 来自 checkpoint；filelist 见上（split=train 时与训练同路径）；MyMuscleDataset(step, std, cond, percent) 同训练
    # - 切片：_prepare_eval_batch 与 _prepare_batch 一致，均取前 L 帧（无随机 crop）
    # - Loss：model(..., bypass_vq=False) 返回的 stats["joints3d/self_smooth_l1"] 等与训练同公式（F.smooth_l1_loss(pred, target), reduction='mean'）
    # - 归一化：standardizer 从 checkpoint 加载，eval 时 update=False
    print(
        f"[Eval] data 与训练对齐: pack_mode={pack_mode}, clip_len={clip_len}, "
        f"joints3d_root_center={data_cfg.get('joints3d_root_center', True)}, "
        f"step={data_cfg.get('step', 30)}, std={data_cfg.get('std', 'False')}"
    )
    print(
        "[Eval] codebook 统计与 Stage2 对齐: 仅用每段前 step 帧参与 j3d/emg 激活率，避免 padding 帧拉低激活率"
    )

    stage2_model = None
    if stage2_ckpt_path is not None and stage2_ckpt_path.exists():
        import importlib.util
        eval_s2_path = Path(__file__).resolve().parent / "eval_stage2_pose2emg_official_metrics.py"
        spec = importlib.util.spec_from_file_location("eval_stage2_pose2emg_official_metrics", eval_s2_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stage2_model, _, _, _ = mod._build_stage2_from_ckpt(stage2_ckpt_path, device, stage1_override_ckpt=None)
        print("[Eval] codebook 统计将使用 Stage2 模型算 idx_j3d，与 Stage2 评测完全一致")

    from musclesinaction.dataloader.data import MyMuscleDataset, _seed_worker  # type: ignore

    dset = MyMuscleDataset(
        str(filelist_path),
        _NullLogger(),
        split,
        percent=float(data_cfg.get("percent", 1.0)),
        step=int(data_cfg.get("step", 30)),
        std=str(data_cfg.get("std", "False")),
        cond=str(data_cfg.get("cond", "True")),
        transform=None,
    )

    loader = DataLoader(
        dset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
        worker_init_fn=_seed_worker,
        pin_memory=(device.type == "cuda"),
    )

    joints3d_root_center = bool(data_cfg.get("joints3d_root_center", True))
    joints3d_root_index = int(data_cfg.get("joints3d_root_index", 8))
    # 与 Stage2 对齐：codebook 统计仅用每段前 step 帧，避免变长 collate 带来的 padding 帧（零/重复输入只激活少量 code，导致 j3d 激活率偏低）
    step_frames = int(data_cfg.get("step", 30))

    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    num_codes = int(model.cfg.vq.num_codes)
    j3d_global_counts = np.zeros(num_codes, dtype=np.int64)
    emg_global_counts = np.zeros(num_codes, dtype=np.int64)
    j3d_batch_unique_counts: List[int] = []
    emg_batch_unique_counts: List[int] = []
    j3d_batch_perplexities: List[float] = []
    emg_batch_perplexities: List[float] = []
    batch_counter = 0
    sample_counter = 0
    # 与训练同口径：与 model.forward 内 _smooth_l1(pred, target) 完全一致
    # F.smooth_l1_loss(pred, target) 默认 reduction='mean'，即对整张量所有元素求平均。
    # 下面按 clip 数加权累加后除以总 clip 数 = 全验证集上 element-wise Smooth L1 的全局均值。
    recon_accums: Dict[str, float] = {
        "joints3d_self": 0.0,
        "emg_self": 0.0,
        "emg_to_joints3d": 0.0,
        "joints3d_to_emg": 0.0,
    }
    recon_total_samples = 0

    for batch in tqdm(loader, desc="eval", unit="batch"):
        joints3d_flat, emg_flat, b, t = _prepare_eval_batch(
            batch,
            device,
            joints3d_root_center=joints3d_root_center,
            joints3d_root_index=joints3d_root_index,
            pack_mode=pack_mode,
            clip_len=clip_len,
        )
        # 与训练一致：必须为 clip_len 帧一段，TCN 感受野依赖序列长度
        if pack_mode in ("clip", "sequence", "seq"):
            expected_j = int(clip_len) * 75
            expected_e = int(clip_len) * 8
            if joints3d_flat.shape[1] != expected_j or emg_flat.shape[1] != expected_e:
                raise ValueError(
                    f"Eval clip shape 与训练不一致: joints3d_flat {joints3d_flat.shape[1]} != {expected_j}, "
                    f"emg_flat {emg_flat.shape[1]} != {expected_e}. 请保证 data.pack.clip_len={clip_len} 与训练一致。"
                )
        # 与训练同口径：同 model._smooth_l1(pred,target)，同 target=normalize(x, update=False)
        with torch.no_grad():
            _, stats = model(
                x_joints3d=joints3d_flat,
                x_smpl_pose=None,
                x_emg=emg_flat,
                emg_weight=1.0,
                bypass_vq=False,
                loss_w_joints3d_self=1.0,
                loss_w_vq_joints3d=1.0,
                loss_w_emg_self=1.0,
                loss_w_emg_to_joints3d=1.0,
                loss_w_joints3d_to_emg=1.0,
                loss_w_vq_emg=1.0,
            )
        recon_accums["joints3d_self"] += float(stats["joints3d/self_smooth_l1"].item()) * b
        recon_accums["emg_self"] += float(stats["emg/self_smooth_l1"].item()) * b
        recon_accums["emg_to_joints3d"] += float(stats["emg_to_joints3d/smooth_l1"].item()) * b
        recon_accums["joints3d_to_emg"] += float(stats["joints3d_to_emg/smooth_l1"].item()) * b
        recon_total_samples += b

        eval_bypass = getattr(model.cfg, "encoder_decoder_only", False)
        # Codebook 统计：与 Stage2 对齐，仅用每段前 step_frames 帧，避免 padding 帧拉低激活率
        if pack_mode in ("frame", "frames"):
            raw_j = _to_tensor(batch["3dskeleton"]).to(device=device, dtype=torch.float32)
            raw_e = _to_tensor(batch["emg_values"]).to(device=device, dtype=torch.float32)
            if joints3d_root_center:
                raw_j = _root_center_joints3d(raw_j, joints3d_root_index)
            b_raw, t_raw, _, _ = raw_j.shape
            L = min(step_frames, t_raw)
            joints3d_flat_for_codes = raw_j[:, :L].contiguous().reshape(b_raw * L, 75)
            emg_flat_for_codes = raw_e[:, :, :L].permute(0, 2, 1).contiguous().reshape(b_raw * L, 8)
            # 首 batch 诊断：若 t_raw==L 且与 full 一致，说明 val 全是 step 帧，截断无效果
            if batch_counter == 0:
                same_shape = b * t == b_raw * L
                same_data = False
                if same_shape and joints3d_flat.shape[0] == joints3d_flat_for_codes.shape[0]:
                    same_data = bool(
                        torch.allclose(joints3d_flat, joints3d_flat_for_codes, rtol=0, atol=1e-6)
                    )
                print(
                    f"[Eval] codebook 首 batch: t_raw={t_raw}, L={L}, full_shape={joints3d_flat.shape}, "
                    f"for_codes_shape={joints3d_flat_for_codes.shape}, same_data={same_data}"
                )
        else:
            joints3d_flat_for_codes = joints3d_flat
            emg_flat_for_codes = emg_flat

        pred_emg_flat, idx_j, ppl_j, _ = _predict_emg_from_joints_raw(
            model, joints3d_flat, bypass_vq=eval_bypass
        )
        _, _, idx_e_full, _, ppl_e, _ = model._encode_quantize(
            model.emg, emg_flat, bypass_vq=eval_bypass
        )
        # 仅用前 step 帧的编码结果做 codebook 统计（与 Stage2 口径一致）；若提供了 --stage2_ckpt 则用 Stage2 模型算 idx_j 与 Stage2 评测完全一致
        if stage2_model is not None:
            j3d_stage2 = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
            if joints3d_root_center:
                j3d_stage2 = _root_center_joints3d(j3d_stage2, joints3d_root_index)
            with torch.no_grad():
                out_s2 = stage2_model(j3d_stage2)
            idx_j = out_s2["idx_j3d"]  # (B, T, 63)
            _ij = idx_j.detach().cpu().numpy().reshape(-1)
            _cnt = np.bincount(_ij, minlength=num_codes).astype(np.float64)
            _p = _cnt / max(float(_cnt.sum()), 1.0)
            _p = _p[_p > 0]
            ppl_j_codes = torch.tensor(
                float(np.exp(-np.sum(_p * np.log(_p)))) if _p.size > 0 else 0.0,
                device=device,
            )
            _, _, idx_e, _, ppl_e_codes, _ = model._encode_quantize(
                model.emg, emg_flat_for_codes, bypass_vq=eval_bypass
            )
        else:
            _, _, idx_j, _, ppl_j_codes, _ = model._encode_quantize(
                model.joints3d, joints3d_flat_for_codes, bypass_vq=eval_bypass
            )
            _, _, idx_e, _, ppl_e_codes, _ = model._encode_quantize(
                model.emg, emg_flat_for_codes, bypass_vq=eval_bypass
            )

        idx_j_np = idx_j.detach().cpu().numpy().reshape(-1)
        idx_e_np = idx_e.detach().cpu().numpy().reshape(-1)
        j3d_global_counts += np.bincount(idx_j_np, minlength=num_codes)
        emg_global_counts += np.bincount(idx_e_np, minlength=num_codes)
        j3d_batch_unique_counts.append(int(np.unique(idx_j_np).size))
        emg_batch_unique_counts.append(int(np.unique(idx_e_np).size))
        j3d_batch_perplexities.append(float(ppl_j_codes.item()))
        emg_batch_perplexities.append(float(ppl_e_codes.item()))

        pred_bt8 = pred_emg_flat.view(b, t, 8).detach().cpu().numpy()
        gt_bt8 = emg_flat.view(b, t, 8).detach().cpu().numpy()

        pred_all.append(pred_bt8.reshape(b * t, 8))
        gt_all.append(gt_bt8.reshape(b * t, 8))

        filepaths = _get_filepaths_from_batch(batch, b)
        for i in range(b):
            seq_sq_err = np.square(pred_bt8[i] - gt_bt8[i])
            seq_abs_err = np.abs(pred_bt8[i] - gt_bt8[i])
            seq_mse = float(np.mean(seq_sq_err))
            seq_rmse = float(math.sqrt(seq_mse))
            seq_mae = float(np.mean(seq_abs_err))
            row = {
                "eval_index": sample_counter,
                "filepath": filepaths[i],
                "frames": t,
                "mse": seq_mse,
                "rmse": seq_rmse,
                "mae": seq_mae,
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

    # 与训练同口径：自重建/交叉重建 Smooth L1 均值（按样本数加权）
    recon_avg = (
        {k: recon_accums[k] / recon_total_samples for k in recon_accums}
        if recon_total_samples > 0
        else {k: float("nan") for k in recon_accums}
    )

    run_dir = out_root / ckpt_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sequence_csv = run_dir / "per_sequence.csv"
    summary_json = run_dir / "summary.json"
    pred_npz = run_dir / "pred_vs_gt.npz"
    codebook_json = run_dir / "codebook_usage.json"

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
        "checkpoint": str(ckpt_path),
        "config_path_fallback": str(fallback_config_path),
        "split": split,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_samples": None if max_samples is None else int(max_samples),
        "emg_online_std": bool(emg_online_std),
        "eval_in_train_mode": bool(eval_in_train_mode),
        "num_batches": int(batch_counter),
        "num_sequences": int(len(per_sequence_rows)),
        **metrics,
        "recon_loss_joints3d_self": recon_avg["joints3d_self"],
        "recon_loss_emg_self": recon_avg["emg_self"],
        "recon_loss_emg_to_joints3d": recon_avg["emg_to_joints3d"],
        "recon_loss_joints3d_to_emg": recon_avg["joints3d_to_emg"],
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for the custom frame codebook using official-like pose->EMG metrics."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="custom/configs/frame_codebook_mia_train.yaml",
        help="Fallback YAML config when checkpoint does not contain an embedded config.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path to evaluate.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing *.pt checkpoints. If omitted, use custom/checkpoints/frame_codebook.",
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="If set, evaluate every *.pt checkpoint under --checkpoint_dir; otherwise evaluate only the latest.",
    )
    parser.add_argument("--split", type=str, default="val", help="train / val / test. If test missing, fallback to val.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cpu', 'cuda' (default GPU), or 'cuda:N' for GPU index N (e.g. cuda:0, cuda:1).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of sequence samples.")
    parser.add_argument(
        "--emg_online_std",
        type=_str2bool,
        default=True,
        help="Whether to enable EMG online standardization during evaluation. Default: true.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: custom/output/eval_frame_codebook_official/<timestamp>/",
    )
    parser.add_argument(
        "--eval_in_train_mode",
        action="store_true",
        help="Run evaluation with model.train() instead of model.eval(). 试金石：若 Loss 从异常高降回正常，则多半是 OnlineStandardizer 的 running stats 在 eval 时有问题。",
    )
    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        default=None,
        help="Optional Stage2 checkpoint path. 若提供，codebook 的 j3d 激活率将用 Stage2 模型算 idx_j3d，与 Stage2 评测完全一致（可复现 100%% 激活率）。",
    )
    args = parser.parse_args()

    mia_root = get_musclesinaction_repo_root()
    split = _resolve_split(args.split, mia_root)
    fallback_config_path = Path(args.config).expanduser().resolve()
    ckpt_paths = _collect_checkpoints(
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        all_checkpoints=bool(args.all_checkpoints),
        mia_root=mia_root,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_root = Path(args.out_dir).expanduser().resolve()
    else:
        out_root = (mia_root / "custom" / "output" / "eval_frame_codebook_official" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    all_rows: List[Dict[str, Any]] = []
    all_json = out_root / "summary_all_checkpoints.csv"

    print("[Eval] split =", split)
    print("[Eval] device =", device)
    print("[Eval] out_root =", out_root)
    print("[Eval] num_checkpoints =", len(ckpt_paths))
    if args.eval_in_train_mode:
        print("[Eval] eval_in_train_mode = True（model.train() 跑评测，试金石排查 OnlineStandardizer）")

    stage2_ckpt_path = None
    if getattr(args, "stage2_ckpt", None):
        p = Path(args.stage2_ckpt).expanduser()
        if not p.is_absolute():
            p = (mia_root / p).resolve()
        stage2_ckpt_path = p

    for ckpt_path in ckpt_paths:
        print("[Eval] checkpoint =", ckpt_path)
        summary = _evaluate_one_checkpoint(
            ckpt_path=ckpt_path,
            fallback_config_path=fallback_config_path,
            split=split,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_samples=args.max_samples,
            out_root=out_root,
            emg_online_std=bool(args.emg_online_std),
            eval_in_train_mode=bool(args.eval_in_train_mode),
            stage2_ckpt_path=stage2_ckpt_path,
        )
        row = {
            "checkpoint": summary["checkpoint"],
            "split": summary["split"],
            "eval_in_train_mode": summary["eval_in_train_mode"],
            "num_sequences": summary["num_sequences"],
            "num_frames": summary["num_frames"],
            "official_global_mse": summary["official_global_mse"],
            "official_global_rmse": summary["official_global_rmse"],
            "official_global_mae": summary["official_global_mae"],
            "recon_loss_joints3d_self": summary["recon_loss_joints3d_self"],
            "recon_loss_emg_self": summary["recon_loss_emg_self"],
            "recon_loss_emg_to_joints3d": summary["recon_loss_emg_to_joints3d"],
            "recon_loss_joints3d_to_emg": summary["recon_loss_joints3d_to_emg"],
            "j3d_active_codes": summary["codebook_usage"]["joints3d"]["active_codes"],
            "j3d_active_rate": summary["codebook_usage"]["joints3d"]["active_rate"],
            "j3d_avg_batch_unique_codes": summary["codebook_usage"]["joints3d"]["avg_batch_unique_codes"],
            "emg_active_codes": summary["codebook_usage"]["emg"]["active_codes"],
            "emg_active_rate": summary["codebook_usage"]["emg"]["active_rate"],
            "emg_avg_batch_unique_codes": summary["codebook_usage"]["emg"]["avg_batch_unique_codes"],
            "summary_json": str((out_root / Path(ckpt_path).stem / "summary.json").resolve()),
        }
        all_rows.append(row)
        print(
            "[EvalResult]",
            {
                "checkpoint": ckpt_path.name,
                "eval_in_train_mode": summary["eval_in_train_mode"],
                "official_global_rmse": summary["official_global_rmse"],
                "official_global_mse": summary["official_global_mse"],
                "num_sequences": summary["num_sequences"],
                "recon_loss_joints3d_self": summary["recon_loss_joints3d_self"],
                "recon_loss_emg_self": summary["recon_loss_emg_self"],
                "recon_loss_emg_to_joints3d": summary["recon_loss_emg_to_joints3d"],
                "recon_loss_joints3d_to_emg": summary["recon_loss_joints3d_to_emg"],
                "j3d_active_codes": summary["codebook_usage"]["joints3d"]["active_codes"],
                "j3d_active_rate": summary["codebook_usage"]["joints3d"]["active_rate"],
                "emg_active_codes": summary["codebook_usage"]["emg"]["active_codes"],
                "emg_active_rate": summary["codebook_usage"]["emg"]["active_rate"],
            },
        )

    _write_csv_rows(
        all_json,
        all_rows,
        [
            "checkpoint",
            "split",
            "eval_in_train_mode",
            "num_sequences",
            "num_frames",
            "official_global_mse",
            "official_global_rmse",
            "official_global_mae",
            "recon_loss_joints3d_self",
            "recon_loss_emg_self",
            "recon_loss_emg_to_joints3d",
            "recon_loss_joints3d_to_emg",
            "j3d_active_codes",
            "j3d_active_rate",
            "j3d_avg_batch_unique_codes",
            "emg_active_codes",
            "emg_active_rate",
            "emg_avg_batch_unique_codes",
            "summary_json",
        ],
    )
    print("[Eval] saved_summary_csv =", all_json)


if __name__ == "__main__":
    main()
