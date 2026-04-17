from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# File path:
#   .../golf_third_party/musclesinaction/custom/train/train_frame_codebook.py
_MIA_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.models.frame_codebook import FrameCodebookConfig, FrameCodebookModel, ModalityConfig  # noqa: E402
from custom.models.temporal import TemporalConfig  # noqa: E402
from custom.models.vq_ema import VQEMAConfig  # noqa: E402
from custom.utils.amass_filelist import build_amass_filelist  # noqa: E402
from custom.utils.mia_filelist import build_mia_train_filelist  # noqa: E402
from custom.utils.online_standardize import OnlineStandardizeConfig  # noqa: E402
from custom.utils.path_utils import get_musclesinaction_repo_root  # noqa: E402


class _NullLogger:
    def info(self, *args, **kwargs):
        return

    def warning(self, *args, **kwargs):
        return

    def exception(self, *args, **kwargs):
        return


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


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


def _get_lr_warmup_cosine(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """LR = linear warmup then cosine decay to min_lr. step is 0-based."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps if warmup_steps > 0 else base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _vq_commitment_beta_schedule(step: int, max_steps: int, runtime_cfg: Dict[str, Any]) -> Optional[float]:
    """
    三阶段 VQ commitment 权重 β(t)（仅训练、且未 bypass_vq 时传入 VQ）：
    - 阶段一 [0, phase1_frac·T)：β = beta_free（自由探索）
    - 阶段二 [phase1_frac·T, phase2_end_frac·T)：线性或余弦升到 beta_target（温和收网）
    - 阶段三 [phase2_end_frac·T, T]：β = beta_target（稳定微调）
    若 enabled=false 则返回 None，VQ 使用 model.vq.cfg.beta。
    """
    sched = runtime_cfg.get("vq_beta_schedule")
    if not isinstance(sched, dict) or not bool(sched.get("enabled", False)):
        return None
    T = int(sched.get("total_steps") or max_steps)
    if T <= 0:
        return None
    p1 = float(sched.get("phase1_frac", 0.1))
    p2 = float(sched.get("phase2_end_frac", 0.5))
    beta_free = float(sched.get("beta_free", 0.0))
    beta_target = float(sched.get("beta_target", 0.25))
    ramp = str(sched.get("ramp", "linear")).lower().strip()
    s = float(step)
    t1 = p1 * T
    t2 = p2 * T
    if s < t1:
        return beta_free
    if t2 <= t1 + 1e-12:
        return beta_target
    if s < t2:
        u = (s - t1) / (t2 - t1)
        u = max(0.0, min(1.0, u))
        if ramp in ("cosine", "cos"):
            w = 0.5 * (1.0 - math.cos(math.pi * u))
        else:
            w = u
        return beta_free + (beta_target - beta_free) * w
    return beta_target


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(str(tmp), str(path))
        return True
    except Exception as e1:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        try:
            torch.save(payload, tmp, _use_new_zipfile_serialization=False)
            os.replace(str(tmp), str(path))
            return True
        except Exception as e2:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            print(f"[Checkpoint] save failed: {path} | err1={type(e1).__name__}: {e1} | err2={type(e2).__name__}: {e2}")
            return False


def _codebook_kmeans_init(
    model: FrameCodebookModel,
    loader: DataLoader,
    device: torch.device,
    target_vectors: int,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    pack_mode: str,
    clip_len: int,
    runtime_cfg: Dict[str, Any],
) -> None:
    """
    After stage1 (warmup_1): collect encoder outputs from batches, run k-means,
    and set codebook embedding (and EMA buffers) to cluster centers.
    """
    model.eval()
    k = int(model.vq.cfg.num_codes)
    d = int(model.vq.cfg.code_dim)
    vectors: List[torch.Tensor] = []
    n_batches = 0
    it = iter(loader)
    print(f"[Codebook k-means init] target vectors={target_vectors}, k={k}, code_dim={d}")
    while sum(v.shape[0] for v in vectors) < target_vectors:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        joints3d_f, _, emg_f = _prepare_batch(
            batch,
            device,
            joints3d_root_center=joints3d_root_center,
            joints3d_root_index=joints3d_root_index,
            pack_mode=pack_mode,
            clip_len=clip_len,
        )
        with torch.no_grad():
            out = model.get_encoder_outputs_for_init(
                x_joints3d=joints3d_f,
                x_smpl_pose=None,
                x_emg=emg_f,
            )
        for name, z in out.items():
            # z: (B, token_count, code_dim) -> (B*T, code_dim)
            z_flat = z.reshape(-1, d)
            vectors.append(z_flat)
        n_batches += 1
    stacked = torch.cat(vectors, dim=0)
    n_total = int(stacked.shape[0])
    if n_total > target_vectors:
        stacked = stacked[:target_vectors]
        n_total = target_vectors
    print(f"[Codebook k-means init] collected {n_total} vectors from {n_batches} batches, running KMeans(k={k})...")
    X = stacked.cpu().numpy()
    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=300, random_state=42)
    kmeans.fit(X)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(dtype=model.vq.embedding.dtype, device=device)
    cnorm = centers.norm(dim=1).mean().item()
    virtual = float(runtime_cfg.get("codebook_kmeans_ema_virtual_per_code", 100.0))
    # embedding + EMA: pretend each code saw `virtual` samples at its k-means center (global prior).
    with torch.no_grad():
        model.vq.embedding.copy_(centers)
        model.vq.ema_cluster_size.fill_(virtual)
        model.vq.ema_embed_sum.copy_(centers * virtual)
    print(
        f"[Codebook k-means init] done. Centers norm mean={cnorm:.4f}, "
        f"ema virtual_mass_per_code={virtual}"
    )
    model.train()


def _apply_runtime_overrides(cfg: Dict[str, Any], *, emg_online_std: bool) -> Dict[str, Any]:
    model_cfg = cfg.setdefault("model", {})
    modalities = model_cfg.setdefault("modalities", {})
    emg_cfg = modalities.setdefault("emg", {})
    emg_cfg["online_std"] = bool(emg_online_std)
    
    # Auto-compute in_dim and seq_len for clip mode
    data_cfg = cfg.get("data", {})
    pack_cfg = data_cfg.get("pack", {})
    pack_mode = str(pack_cfg.get("mode", "frame")).strip().lower()
    clip_len = int(pack_cfg.get("clip_len", data_cfg.get("step", 30)))
    
    if pack_mode in ("clip", "sequence", "seq"):
        for mod_name, base_dim in [("joints3d", 75), ("emg", 8), ("smpl_pose", 72)]:
            if mod_name in modalities and modalities[mod_name] is not None:
                mod = modalities[mod_name]
                mod["in_dim"] = clip_len * base_dim
                if "temporal" not in mod:
                    mod["temporal"] = {}
                mod["temporal"]["seq_len"] = clip_len
                
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


def _build_model(cfg: Dict[str, Any], device: torch.device) -> FrameCodebookModel:
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


def _root_center_joints3d(joints3d: torch.Tensor, root_index: int) -> torch.Tensor:
    if joints3d.ndim != 4 or joints3d.shape[-2:] != (25, 3):
        raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
    if not (0 <= int(root_index) < int(joints3d.shape[2])):
        raise ValueError(f"root_index out of range: {root_index}")
    root = joints3d[:, :, root_index : root_index + 1, :]
    return joints3d - root


def _prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    pack_mode: str = "frame",
    clip_len: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Official dataloader returns:
      - 3dskeleton: (B, T, 25, 3)
      - pose:       (B, T, 72)
      - emg_values: (B, 8, T)
    MIADatasetOfficial: each sample (one __getitem__) is one sequence; native length is 30 frames.
    Two packing modes:
      - frame: flatten over frames and train a frame-wise codebook
          joints3d_flat: (B*T, 75), emg_flat: (B*T, 8)  [B sequences, T frames each]
      - clip: one sequence = one codebook input (one load = 30 frames per sample)
          joints3d_clip: (B, clip_len*75), emg_clip: (B, clip_len*8)  [B samples, each 30 frames]
    """
    joints3d = _to_tensor(batch["3dskeleton"]).to(device=device, dtype=torch.float32)
    emg = _to_tensor(batch["emg_values"]).to(device=device, dtype=torch.float32)

    if joints3d.ndim != 4:
        raise ValueError(f"Expected 3dskeleton (B,T,25,3), got {tuple(joints3d.shape)}")
    if emg.ndim != 3:
        raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")

    b, t, j, c = joints3d.shape
    if j != 25 or c != 3:
        raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
    if emg.shape != (b, 8, t):
        raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")

    if joints3d_root_center:
        joints3d = _root_center_joints3d(joints3d, root_index=joints3d_root_index)

    mode = str(pack_mode).lower().strip()
    if mode in ("frame", "frames"):
        joints3d_flat = joints3d.reshape(b * t, 75)
        emg_flat = emg.permute(0, 2, 1).contiguous().reshape(b * t, 8)
        return joints3d_flat, torch.empty((0, 72), device=device), emg_flat

    if mode in ("clip", "sequence", "seq"):
        # 将原始序列划分为长度为 clip_len 的多个子片段，如果无法整除，则丢弃末尾部分
        L = int(clip_len)
        if t < L:
            raise ValueError(f"Clip length mismatch: got T={t} < clip_len={L}")
        
        num_clips = t // L
        joints3d = joints3d[:, :num_clips * L]
        emg = emg[:, :, :num_clips * L]

        # Reshape into (B * num_clips, L, ...)
        joints3d_clip = joints3d.reshape(b * num_clips, L, 25, 3).contiguous().reshape(b * num_clips, L * 75)
        
        # emg: (B, 8, num_clips * L) -> (B, num_clips, L, 8) -> (B * num_clips, L, 8)
        emg_clip = emg.permute(0, 2, 1).contiguous().reshape(b, num_clips, L, 8).reshape(b * num_clips, L * 8)
        return joints3d_clip, torch.empty((0, 72), device=device), emg_clip

    raise ValueError(f"Unsupported data.pack.mode: {pack_mode}")


@torch.no_grad()
def _eval_one_epoch(
    model: FrameCodebookModel,
    loader: DataLoader,
    device: torch.device,
    *,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    pack_mode: str,
    clip_len: int,
    loss_weights: Dict[str, float],
) -> float:
    """跑一轮 val，返回平均 total_loss（与训练同口径，用于按 val_loss 保存 best.pt）。"""
    model.eval()
    loss_list: List[float] = []
    for batch in loader:
        joints3d_f, _, emg_f = _prepare_batch(
            batch,
            device,
            joints3d_root_center=joints3d_root_center,
            joints3d_root_index=joints3d_root_index,
            pack_mode=pack_mode,
            clip_len=clip_len,
        )
        _, stats = model(
            x_joints3d=joints3d_f,
            x_smpl_pose=None,
            x_emg=emg_f,
            emg_weight=loss_weights.get("emg_branch", 1.0),
            bypass_vq=False,
            loss_w_joints3d_self=loss_weights.get("joints3d_self", 1.0),
            loss_w_vq_joints3d=loss_weights.get("vq_joints3d", 1.0),
            loss_w_emg_self=loss_weights.get("emg_self", 1.0),
            loss_w_emg_to_joints3d=loss_weights.get("emg_to_joints3d", 1.0),
            loss_w_joints3d_to_emg=loss_weights.get("joints3d_to_emg", 1.0),
            loss_w_vq_emg=loss_weights.get("vq_emg", 1.0),
        )
        loss_list.append(float(stats["total_loss"].item()))
    model.train()
    return sum(loss_list) / max(len(loss_list), 1)


# ---------- 终极试金石：可视化 Loss 的 target（与 vis 脚本同款反归一化+骨架） ----------
BODY25_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14), (0, 15), (15, 17), (0, 16), (16, 18), (14, 19), (19, 20),
    (14, 21), (11, 22), (22, 23), (11, 24),
)


def _denormalize_from_modality(mod, x_n: torch.Tensor) -> torch.Tensor:
    """与 visualize_joints3d_codebook_recon 中完全一致：仅当有 standardizer 时做 x_n*std+mean."""
    stdzr = getattr(mod, "standardizer", None)
    if stdzr is None:
        return x_n
    std = torch.sqrt(torch.clamp_min(stdzr.var, stdzr.cfg.clip_std_min**2) + float(stdzr.cfg.eps))
    return x_n * std + stdzr.mean


def _set_equal_3d_axes(ax, pts_a: np.ndarray, pts_b: np.ndarray) -> None:
    pts = np.concatenate([pts_a, pts_b], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_skeleton(ax, pts: np.ndarray, title: str, color_pts: str, color_edges: str) -> None:
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color_pts, s=22, depthshade=False)
    for i, j in BODY25_EDGES:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color_edges,
            linewidth=1.5,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


@torch.no_grad()
def _visualize_loss_target_vs_gt(
    xj_n: torch.Tensor,
    x_joints3d: torch.Tensor,
    mod_joints3d,
    out_dir: Path,
    step: int,
    clip_len: int = 30,
    model=None,
    one_emg: Optional[torch.Tensor] = None,
) -> None:
    """
    画 GT / Target(denorm) / Pred(denorm)。当 online_std=false 时 Target 与 GT 相同属正常。
    xj_n / x_joints3d: (1, clip_len*75)。若提供 model+one_emg 则再画 Pred（关节自重建）与 GT+Pred overlay。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_denorm = _denormalize_from_modality(mod_joints3d, xj_n)
    L = int(clip_len)
    frame_idx = L // 2

    gt = x_joints3d[:1].view(1, L, 25, 3).contiguous()
    tg = target_denorm[:1].view(1, L, 25, 3).contiguous()

    gt_f = gt[0, frame_idx].detach().cpu().numpy()
    tg_f = tg[0, frame_idx].detach().cpu().numpy()

    pred_f = None
    if model is not None and one_emg is not None:
        was_training = model.training
        model.eval()
        mod = model.joints3d
        tc, cd = int(mod.cfg.token_count), int(mod.cfg.code_dim)
        z_e = mod.encoder(xj_n).view(1, tc, cd)
        if getattr(model.cfg, "encoder_decoder_only", False):
            z_q = z_e.view(1, tc * cd)
        else:
            z_q, _, _, _, _ = model.vq(z_e.reshape(1 * tc, cd))
            z_q = z_q.view(1, tc * cd)
        pred_j_n = mod.decoder(z_q)
        pred_denorm = _denormalize_from_modality(mod, pred_j_n)
        pred_denorm = pred_denorm.view(1, L, 25, 3)
        pred_f = pred_denorm[0, frame_idx].cpu().numpy()
        if was_training:
            model.train()

    if pred_f is not None:
        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 4, 1, projection="3d")
        ax2 = fig.add_subplot(1, 4, 2, projection="3d")
        ax3 = fig.add_subplot(1, 4, 3, projection="3d")
        ax4 = fig.add_subplot(1, 4, 4, projection="3d")
        _draw_skeleton(ax1, gt_f, "GT (input)", "#1f77b4", "#1f77b4")
        _draw_skeleton(ax2, tg_f, "Target (denorm xj_n)", "#d62728", "#d62728")
        _draw_skeleton(ax3, pred_f, "Pred (denorm recon)", "#2ca02c", "#2ca02c")
        _draw_skeleton(ax4, gt_f, "Overlay: GT + Pred", "#1f77b4", "#1f77b4")
        _draw_skeleton(ax4, pred_f, "Overlay: GT + Pred", "#2ca02c", "#2ca02c")
        ax4.set_title("Overlay: GT (blue) + Pred (green)")
        all_pts = np.concatenate([gt_f, tg_f, pred_f], axis=0)
        for ax in (ax1, ax2, ax3, ax4):
            _set_equal_3d_axes(ax, all_pts, all_pts)
        fig.suptitle(f"Step {step} — GT / Target / Pred (frame {frame_idx})")
    else:
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        _draw_skeleton(ax1, gt_f, "GT (input)", "#1f77b4", "#1f77b4")
        _draw_skeleton(ax2, tg_f, "Target (denorm xj_n)", "#d62728", "#d62728")
        _draw_skeleton(ax3, gt_f, "Overlay: GT + Target", "#1f77b4", "#1f77b4")
        _draw_skeleton(ax3, tg_f, "Overlay: GT + Target", "#d62728", "#d62728")
        ax3.set_title("Overlay: GT (blue) + Target (red)")
        _set_equal_3d_axes(ax1, gt_f, tg_f)
        _set_equal_3d_axes(ax2, gt_f, tg_f)
        _set_equal_3d_axes(ax3, gt_f, tg_f)
        fig.suptitle(f"Step {step} — GT / Target (frame {frame_idx})")

    fig.tight_layout()
    out_path = out_dir / f"step_{step:07d}_target_vs_gt.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[TargetViz] saved {out_path}")


def _create_log_run_dir(mia_root: Path, experiment_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (mia_root / "custom" / "logs" / f"{experiment_name}_{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="custom/configs/frame_codebook_mia_train.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--emg_online_std",
        type=_str2bool,
        default=True,
        help="Whether to enable EMG online standardization. Default: true.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device: 'cpu', 'cuda', or 'cuda:N' (e.g. cuda:1). Default: use config runtime.device.",
    )
    parser.add_argument(
        "--rebuild-filelist",
        action="store_true",
        help="强制根据当前 config 重新生成 train filelist，不复用已有文件（避免曾用 max_samples=10 等生成的小 filelist 被一直复用导致只训 10 条样本）. Default: False.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(config_path)
    cfg = _apply_runtime_overrides(cfg, emg_online_std=bool(args.emg_online_std))

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    _set_seed(seed)

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))

    # Official dataloader expects cwd = musclesinaction repo root.
    os.chdir(str(mia_root))

    runtime_cfg = cfg["runtime"]
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimizer"]
    data_source = str(data_cfg.get("source", "mia")).strip().lower()

    pack_cfg = (data_cfg.get("pack", {}) or {}) if isinstance(data_cfg, dict) else {}
    pack_mode = str(pack_cfg.get("mode", "frame")).strip().lower()
    clip_len = int(pack_cfg.get("clip_len", data_cfg.get("step", 30)))

    device_str = args.device if args.device is not None else str(runtime_cfg.get("device", "cuda"))
    if device_str == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
        # Enable CuDNN benchmark for dynamic/large tensor operations (especially 1D Conv) to prevent CUDNN_STATUS_NOT_SUPPORTED fallbacks
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


    train_filelist = Path(str(data_cfg["train_filelist"]))
    if not train_filelist.is_absolute():
        train_filelist = (mia_root / train_filelist).resolve()

    if data_source == "amass":
        smpl_model_path = data_cfg.get("smpl_model_path")
        if smpl_model_path is None:
            raise ValueError("data.smpl_model_path required for data.source=amass")
        smpl_model_path = Path(smpl_model_path)
        if not smpl_model_path.is_absolute():
            smpl_model_path = (mia_root / smpl_model_path).resolve()
        if not train_filelist.exists() or train_filelist.stat().st_size == 0:
            amass_root = data_cfg.get("amass_root")
            if amass_root is None:
                amass_root = mia_root / ".." / ".." / "amass"
            amass_root = Path(amass_root)
            if not amass_root.is_absolute():
                amass_root = (mia_root / amass_root).resolve()
            res = build_amass_filelist(
                amass_root=amass_root,
                out_txt=train_filelist,
                max_files=data_cfg.get("max_samples"),
            )
            print(f"[FileList] AMASS wrote {res.num_files} files -> {res.filelist_path}")
        else:
            print(f"[FileList] reuse existing: {train_filelist}")

        from custom.dataset.amass_dataset import AmassCodebookDataset  # noqa: E402
        from musclesinaction.dataloader.data import _seed_worker  # type: ignore

        dset = AmassCodebookDataset(
            str(train_filelist),
            step=int(data_cfg.get("step", 30)),
            smpl_model_path=str(smpl_model_path),
            joints3d_root_center=bool(data_cfg.get("joints3d_root_center", True)),
            joints3d_root_index=int(data_cfg.get("joints3d_root_index", 8)),
            percent=float(data_cfg.get("percent", 1.0)),
            max_samples=data_cfg.get("max_samples"),
        )
    else:
        force_rebuild = getattr(args, "rebuild_filelist", False)
        if force_rebuild or not train_filelist.exists() or train_filelist.stat().st_size == 0:
            if force_rebuild:
                print("[FileList] --rebuild-filelist: 强制重新生成，忽略已有文件")
            res = build_mia_train_filelist(
                mia_repo_root=mia_root,
                split=str(data_cfg.get("split", "train")),
                out_txt=train_filelist,
                max_samples=data_cfg.get("max_samples", None),
                require_files=data_cfg.get("require_files", None),
            )
            print(f"[FileList] wrote {res.num_samples} samples -> {res.filelist_path}")
        else:
            with open(train_filelist, encoding="utf-8") as f:
                n_lines = sum(1 for line in f if line.strip())
            print(f"[FileList] reuse existing: {train_filelist} ({n_lines} samples)")
            max_samples_cfg = data_cfg.get("max_samples")
            if max_samples_cfg is None and n_lines <= 50:
                print(
                    f"[FileList] WARNING: 当前 config 未限制 max_samples（应用全量数据），但已有 filelist 仅 {n_lines} 条。"
                    " 若你期望用全量数据，请删除该文件后重跑，或加 --rebuild-filelist 强制按当前 config 重建。"
                )

        from musclesinaction.dataloader.data import MyMuscleDataset, _seed_worker  # type: ignore

        dset = MyMuscleDataset(
            str(train_filelist),
            _NullLogger(),
            str(data_cfg.get("split", "train")),
            percent=float(data_cfg.get("percent", 1.0)),
            step=int(data_cfg.get("step", 30)),
            std=str(data_cfg.get("std", "False")),
            cond=str(data_cfg.get("cond", "True")),
            transform=None,
        )

    _bs = int(data_cfg.get("batch_size", 8))
    _n = len(dset)
    _drop_last = bool(data_cfg.get("drop_last", True))
    # drop_last=True 且 len(dset) < batch_size 时，每轮 0 个 batch → next(it) 立刻 StopIteration
    if _n == 0:
        raise RuntimeError(
            f"Train dataset is empty (filelist / filters). Cannot train. filelist={train_filelist}"
        )
    if _drop_last and _n < _bs:
        print(
            f"[DataLoader] len(dset)={_n} < batch_size={_bs}, drop_last=True would yield 0 batches "
            f"→ using drop_last=False (effective batch < {_bs} per step until you add data or lower batch_size)."
        )
        _drop_last = False
    loader = DataLoader(
        dset,
        batch_size=_bs,
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 4)),
        drop_last=_drop_last,
        worker_init_fn=_seed_worker,
        pin_memory=(device.type == "cuda"),
    )

    val_loader: Optional[DataLoader] = None
    if data_cfg.get("source", "mia") == "mia":
        val_filelist = Path(str(data_cfg.get("val_filelist", ""))) if data_cfg.get("val_filelist", None) else None
        if val_filelist is None or str(val_filelist).strip() == "":
            val_filelist = mia_root / "custom" / "tools" / "datasetsplits" / "miaofficial_val_eval.txt"
        val_filelist = Path(str(val_filelist))
        if not val_filelist.is_absolute():
            val_filelist = (mia_root / val_filelist).resolve()
        if not val_filelist.exists() or val_filelist.stat().st_size == 0:
            build_mia_train_filelist(
                mia_repo_root=mia_root,
                split="val",
                out_txt=val_filelist,
                require_files=data_cfg.get("require_files", ["emgvalues.npy", "joints3d.npy"]),
            )
            print(f"[FileList] val 列表不存在，已生成 -> {val_filelist}")
        val_dset = MyMuscleDataset(
            str(val_filelist),
            _NullLogger(),
            "val",
            percent=float(data_cfg.get("percent", 1.0)),
            step=int(data_cfg.get("step", 30)),
            std=str(data_cfg.get("std", "False")),
            cond=str(data_cfg.get("cond", "True")),
            transform=None,
        )
        val_loader = DataLoader(
            val_dset,
            batch_size=_bs,
            shuffle=False,
            num_workers=int(runtime_cfg.get("num_workers", 4)),
            drop_last=False,
            worker_init_fn=_seed_worker,
            pin_memory=(device.type == "cuda"),
        )
        print(f"[Val] filelist={val_filelist}, samples={len(val_dset)}")

    # 3D 关节在线归一化开关：打开后 joints3d 使用 OnlineStandardizer(均值0方差1)，
    # 归一化参数随 state_dict 保存，推理时自动加载
    if runtime_cfg.get("joints3d_online_std", False):
        cfg.setdefault("model", {}).setdefault("modalities", {}).setdefault("joints3d", {})["online_std"] = True
        print("[Runtime] joints3d_online_std=true → model.modalities.joints3d.online_std enabled (normalize saved in checkpoint)")

    model = _build_model(cfg, device)

    pretrain_ckpt = cfg.get("pretrain_ckpt")
    if pretrain_ckpt:
        ckpt_path = Path(pretrain_ckpt).expanduser().resolve()
        if not ckpt_path.is_absolute():
            ckpt_path = (mia_root / ckpt_path).resolve()
        if ckpt_path.exists():
            payload = torch.load(ckpt_path, map_location=device)
            state = payload.get("model_state", payload)
            try:
                model.load_state_dict(state, strict=True)
            except Exception as e:
                if "unexpected key" in str(e) or "missing key" in str(e):
                    print(f"[Checkpoint] strict=True 失败（可能为旧 TCN BatchNorm 的 running_*），改用 strict=False 加载 {ckpt_path}")
                    model.load_state_dict(state, strict=False)
                else:
                    raise
            print(f"[Checkpoint] loaded pretrain from {ckpt_path}")
        else:
            print(f"[Checkpoint] pretrain_ckpt not found: {ckpt_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 2e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    ckpt_dir = (mia_root / str(cfg["checkpoints"]["dir"])).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_ckpt_dir = ckpt_dir / "ckpt"
    step_ckpt_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = str(cfg.get("experiment", {}).get("name", config_path.stem))
    log_run_dir = _create_log_run_dir(mia_root, experiment_name)
    temp_train_out_dir = (mia_root / "custom" / "output" / "temp_train").resolve()
    temp_data_dir = (mia_root / "custom" / "output" / "temp_data").resolve()
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    train_data_paths_file = temp_data_dir / "train_data_paths.txt"
    train_data_paths_handle = open(train_data_paths_file, "a", encoding="utf-8")
    seen_data_paths: set[str] = set()
    tb_dir = log_run_dir / "tensorboard"
    csv_path = log_run_dir / "metrics.csv"
    config_dump_path = log_run_dir / "config.yaml"
    with open(config_dump_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    writer = SummaryWriter(log_dir=str(tb_dir))
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer: Optional[csv.DictWriter] = None

    print("[Config] config_path=", config_path)
    print("[Config] device=", device)
    print("[Config] ckpt_dir=", ckpt_dir)
    print("[Config] step_ckpt_dir=", step_ckpt_dir)
    print("[Config] log_dir=", log_run_dir)
    print("[Config] model_cfg=", model.cfg)

    try:
        model.train()
        start_time = time.time()
        max_steps = int(runtime_cfg.get("max_steps", 5000))
        log_every = int(runtime_cfg.get("log_every", 50))
        save_every = int(runtime_cfg.get("save_every", 500))
        warmup_1_steps = int(runtime_cfg.get("warmup_1_steps", runtime_cfg.get("warmup_steps", 0)))
        warmup_2_steps = int(runtime_cfg.get("warmup_2_steps", 0))
        codebook_kmeans_init_samples = int(runtime_cfg.get("codebook_kmeans_init_samples", 0))
        vq_dead_reset_warmup_2 = bool(runtime_cfg.get("vq_dead_reset_warmup_2", False))
        vq_dead_reset_train = bool(runtime_cfg.get("vq_dead_reset_train", True))
        print(
            "[Config] vq_dead_reset warmup_2=",
            vq_dead_reset_warmup_2,
            "train=",
            vq_dead_reset_train,
            "(effective only if model.vq.reset_dead_codes)",
        )
        _vq_explore = float(runtime_cfg.get("vq_explore_noise_std_frac", 0))
        _vq_explore_mx = runtime_cfg.get("vq_explore_noise_train_max_steps")
        print(
            "[Config] vq_explore_noise std_frac=",
            _vq_explore,
            "train_max_steps=",
            _vq_explore_mx,
            "(joint train only; noise on z_e for argmin only)",
        )
        _bs_sched = runtime_cfg.get("vq_beta_schedule")
        if isinstance(_bs_sched, dict) and bool(_bs_sched.get("enabled", False)):
            print(
                "[Config] vq_beta_schedule enabled:",
                {k: _bs_sched.get(k) for k in ("phase1_frac", "phase2_end_frac", "beta_free", "beta_target", "ramp", "total_steps")},
                "max_steps=", max_steps,
            )
        opt_cfg = cfg.get("optimizer", {})
        lr_schedule = str(opt_cfg.get("lr_schedule", "constant")).strip().lower()
        lr_warmup_steps = int(opt_cfg.get("lr_warmup_steps", 0))
        lr_min = float(opt_cfg.get("lr_min", 1e-6))
        base_lr = float(opt_cfg.get("lr", 2e-4))
        step_len = int(data_cfg.get("step", 30))
        batch_size = int(data_cfg.get("batch_size", 8))
        joints3d_root_center = bool(data_cfg.get("joints3d_root_center", True))
        joints3d_root_index = int(data_cfg.get("joints3d_root_index", 8))
        frames_per_sample = int(clip_len) if pack_mode in ("clip", "sequence", "seq") else int(step_len)

        writer.add_text("config/path", str(config_path))
        writer.add_text("config/log_dir", str(log_run_dir))
        writer.add_text("config/yaml", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False))

        step = 0
        best_loss = float("inf")
        best_val_loss = float("inf")
        eval_every = int(runtime_cfg.get("eval_every", 500))
        lw = (cfg.get("model", {}) or {}).get("loss_weights", {}) or {}
        loss_weights = {
            "emg_branch": float(lw.get("emg_branch", cfg.get("emg_weight", 1.0))),
            "joints3d_self": float(lw.get("joints3d_self", 1.0)),
            "vq_joints3d": float(lw.get("vq_joints3d", 1.0)),
            "emg_self": float(lw.get("emg_self", 1.0)),
            "emg_to_joints3d": float(lw.get("emg_to_joints3d", 1.0)),
            "joints3d_to_emg": float(lw.get("joints3d_to_emg", 1.0)),
            "vq_emg": float(lw.get("vq_emg", 1.0)),
        }
        if val_loader is not None and eval_every > 0:
            print(f"[Config] eval_every={eval_every}，best.pt 将按 val_loss 保存")
        else:
            print("[Config] 无 val 或 eval_every<=0，best.pt 将按 train loss 保存")
            
        from tqdm import tqdm
        
        pbar = tqdm(total=log_every, desc="Training", leave=False, file=sys.stdout)
        
        it = iter(loader)
        while step < max_steps:
            if step == warmup_1_steps and codebook_kmeans_init_samples > 0:
                _codebook_kmeans_init(
                    model,
                    loader,
                    device,
                    target_vectors=codebook_kmeans_init_samples,
                    joints3d_root_center=joints3d_root_center,
                    joints3d_root_index=joints3d_root_index,
                    pack_mode=pack_mode,
                    clip_len=clip_len,
                    runtime_cfg=runtime_cfg,
                )

            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            # 记录本 step 参与训练的数据路径到 output/temp_data/train_data_paths.txt，并统计不同样本数
            paths_this_batch = batch.get("filepath")
            if paths_this_batch is not None:
                if isinstance(paths_this_batch, (list, tuple)):
                    pass
                else:
                    paths_this_batch = [paths_this_batch]
                for p in paths_this_batch:
                    path_str = str(p).strip()
                    if path_str and path_str not in seen_data_paths:
                        seen_data_paths.add(path_str)
                    if path_str:
                        train_data_paths_handle.write(path_str + "\n")
                train_data_paths_handle.flush()

            if lr_schedule == "warmup_cosine":
                lr = _get_lr_warmup_cosine(
                    step, lr_warmup_steps, max_steps, base_lr, lr_min
                )
                for g in optimizer.param_groups:
                    g["lr"] = lr
            current_lr = optimizer.param_groups[0]["lr"]

            joints3d_f, _, emg_f = _prepare_batch(
                batch,
                device,
                joints3d_root_center=joints3d_root_center,
                joints3d_root_index=joints3d_root_index,
                pack_mode=pack_mode,
                clip_len=clip_len,
            )
            lw = (cfg.get("model", {}) or {}).get("loss_weights", {}) or {}
            emg_weight = float(lw.get("emg_branch", cfg.get("emg_weight", 1.0)))
            w_j_self = float(lw.get("joints3d_self", 1.0))
            w_vq_j = float(lw.get("vq_joints3d", 1.0))
            w_e_self = float(lw.get("emg_self", 1.0))
            w_etj = float(lw.get("emg_to_joints3d", 1.0))
            w_jte = float(lw.get("joints3d_to_emg", 1.0))
            w_vq_e = float(lw.get("vq_emg", 1.0))
            bypass_vq = step < warmup_1_steps
            freeze_enc_dec = warmup_1_steps <= step < warmup_1_steps + warmup_2_steps
            if freeze_enc_dec:
                model.joints3d.eval()
                model.emg.eval()
                model.vq.train()
                skip_vq_dead_reset = not vq_dead_reset_warmup_2
            else:
                model.train()
                skip_vq_dead_reset = not vq_dead_reset_train
            train_step = step - warmup_1_steps - warmup_2_steps
            if bypass_vq or freeze_enc_dec:
                vq_explore_noise_std_frac = 0.0
            else:
                vq_explore_noise_std_frac = float(runtime_cfg.get("vq_explore_noise_std_frac", 0))
                _mx = runtime_cfg.get("vq_explore_noise_train_max_steps")
                if _mx is not None and train_step >= int(_mx):
                    vq_explore_noise_std_frac = 0.0
            vq_beta_sched = _vq_commitment_beta_schedule(step, max_steps, runtime_cfg)
            vq_commitment_beta = None if bypass_vq else vq_beta_sched
            loss, stats = model(
                x_joints3d=joints3d_f,
                x_smpl_pose=None,
                x_emg=emg_f,
                emg_weight=emg_weight,
                bypass_vq=bypass_vq,
                skip_vq_dead_reset=skip_vq_dead_reset,
                vq_explore_noise_std_frac=vq_explore_noise_std_frac,
                vq_commitment_beta=vq_commitment_beta,
                loss_w_joints3d_self=w_j_self,
                loss_w_vq_joints3d=w_vq_j,
                loss_w_emg_self=w_e_self,
                loss_w_emg_to_joints3d=w_etj,
                loss_w_joints3d_to_emg=w_jte,
                loss_w_vq_emg=w_vq_e,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if freeze_enc_dec:
                vq_param_ids = {id(p) for p in model.vq.parameters()}
                for p in model.parameters():
                    if id(p) not in vq_param_ids and p.grad is not None:
                        p.grad.zero_()
            optimizer.step()

            # Update pbar BEFORE incrementing step, so we count this current step
            # Note: step starts at 0, so after this block it becomes 1. 
            # We want to update the progress bar for every step.
            pbar.update(1)
            step += 1
            
            # 终极试金石：每 100 step 画 GT / Target(denorm) / Pred(denorm)，Overlay GT+Pred
            if step % 100 == 0:
                one_joints3d = joints3d_f[:1]
                one_emg = emg_f[:1]
                xj_n_one = model.joints3d.normalize(one_joints3d, update=False)
                _visualize_loss_target_vs_gt(
                    xj_n_one,
                    one_joints3d,
                    model.joints3d,
                    temp_train_out_dir,
                    step,
                    clip_len=clip_len,
                    model=model,
                    one_emg=one_emg,
                )
            in_warmup = step <= warmup_1_steps + warmup_2_steps
            if step % log_every == 0 or step == 1:
                pbar.close()
                dt = max(time.time() - start_time, 1e-6)
                fps = (step * batch_size * frames_per_sample) / dt
                phase = "warmup_1" if step <= warmup_1_steps else ("warmup_2" if step <= warmup_1_steps + warmup_2_steps else "train")
                beta_eff = float(model.vq.cfg.beta) if vq_commitment_beta is None else float(vq_commitment_beta)
                msg = {
                    "step": step,
                    "loss": float(stats["total_loss"].item()),
                    "lr": current_lr,
                    "phase": phase,
                    "warmup": 1 if in_warmup else 0,
                    "bypass_vq": 1 if bypass_vq else 0,
                    "vq_commitment_beta": beta_eff,
                    "fps_frames_per_sec": float(fps),
                    "vq_resets": int(stats["vq/num_resets"].item()),
                    "j3d_ppl": float(stats["joints3d/perplexity"].item()),
                    "j3d_unique": int(stats["joints3d/codes_unique"].item()),
                    "j3d_self_smooth_l1": float(stats["joints3d/self_smooth_l1"].item()),
                    "j3d_vq_beta_term": float(stats["joints3d/vq_loss"].item()),
                }
                if emg_weight != 0:
                    msg.update({
                        "emg_ppl": float(stats["emg/perplexity"].item()),
                        "emg_unique": int(stats["emg/codes_unique"].item()),
                        "emg_self_smooth_l1": float(stats["emg/self_smooth_l1"].item()),
                        "emg_to_j3d_smooth_l1": float(stats["emg_to_joints3d/smooth_l1"].item()),
                        "j3d_to_emg_smooth_l1": float(stats["joints3d_to_emg/smooth_l1"].item()),
                        "emg_vq_beta_term": float(stats["emg/vq_loss"].item()),
                    })
                print("[Train]", msg)

                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(msg.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(msg)
                csv_file.flush()

                for key, value in msg.items():
                    if key == "step" or not isinstance(value, (int, float)):
                        continue
                    writer.add_scalar(f"train/{key}", value, step)
                writer.flush()

                # 无 val 或 eval_every<=0 时，按 train loss 更新 best.pt；有 val 时仅在下文 eval_every 处按 val_loss 更新
                if not in_warmup and (val_loader is None or eval_every <= 0):
                    current_loss = float(stats["total_loss"].item())
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_path = ckpt_dir / "best.pt"
                        payload = {
                            "step": step,
                            "config_path": str(config_path),
                            "config": cfg,
                            "model_state": model.state_dict(),
                            "optim_state": optimizer.state_dict(),
                        }
                        if _atomic_torch_save(payload, best_path):
                            print(f"[Checkpoint] saved best (train_loss={best_loss:.6f}) -> {best_path}")

            do_eval = (
                not in_warmup
                and val_loader is not None
                and eval_every > 0
                and (step % eval_every == 0 or step == max_steps)
            )
            if do_eval:
                val_loss = _eval_one_epoch(
                    model,
                    val_loader,
                    device,
                    joints3d_root_center=joints3d_root_center,
                    joints3d_root_index=joints3d_root_index,
                    pack_mode=pack_mode,
                    clip_len=clip_len,
                    loss_weights=loss_weights,
                )
                writer.add_scalar("val/total_loss", val_loss, step)
                writer.flush()
                if val_loss < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_path = ckpt_dir / "best.pt"
                    payload = {
                        "step": step,
                        "config_path": str(config_path),
                        "config": cfg,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                    }
                    if _atomic_torch_save(payload, best_path):
                        print(f"[Checkpoint] saved best (val_loss={best_val_loss:.6f}) -> {best_path}")
                        
                        # 保存额外评估指标到 best_metrics.csv
                        try:
                            metrics_csv = ckpt_dir / "best_metrics.csv"
                            j3d_unique = int(stats["joints3d/codes_unique"].item())
                            emg_unique = int(stats["emg/codes_unique"].item()) if emg_weight != 0 else 0
                            k = int(model.vq.cfg.num_codes)
                            act_rate = (j3d_unique + emg_unique) / (2.0 * k) if emg_weight != 0 else j3d_unique / float(k)
                            
                            best_msg = {
                                "step": step,
                                "val_loss": best_val_loss,
                                "activation_rate": f"{act_rate:.4f}",
                                "j3d_self_smooth_l1": float(stats["joints3d/self_smooth_l1"].item()),
                            }
                            if emg_weight != 0:
                                best_msg.update({
                                    "emg_self_smooth_l1": float(stats["emg/self_smooth_l1"].item()),
                                    "emg_to_j3d_smooth_l1": float(stats["emg_to_joints3d/smooth_l1"].item()),
                                    "j3d_to_emg_smooth_l1": float(stats["joints3d_to_emg/smooth_l1"].item()),
                                })
                            
                            write_header = not metrics_csv.exists()
                            with open(metrics_csv, "a", newline="", encoding="utf-8") as bf:
                                bw = csv.DictWriter(bf, fieldnames=list(best_msg.keys()))
                                if write_header:
                                    bw.writeheader()
                                bw.writerow(best_msg)
                        except Exception as e:
                            print(f"[Metrics] Error saving best_metrics.csv: {e}")
                            
                print("[Val]", {"step": step, "val_loss": val_loss, "best_val_loss": float(best_val_loss)})

            if (step % save_every == 0 or step == max_steps) and not in_warmup:
                payload = {
                    "step": step,
                    "config_path": str(config_path),
                    "config": cfg,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                }
                step_path = step_ckpt_dir / f"step_{step:07d}.pt"
                if _atomic_torch_save(payload, step_path):
                    print(f"[Checkpoint] saved step -> {step_path}")
                last_path = ckpt_dir / "last.pt"
                if _atomic_torch_save(payload, last_path):
                    print(f"[Checkpoint] saved last -> {last_path}")
                # 无 val 时在 save_every 也按 train loss 更新 best；有 val 时 best 仅由 eval_every 的 val_loss 更新
                if val_loader is None or eval_every <= 0:
                    current_loss = float(stats["total_loss"].item())
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_path = ckpt_dir / "best.pt"
                        if _atomic_torch_save(payload, best_path):
                            print(f"[Checkpoint] saved best (train_loss={best_loss:.6f}) -> {best_path}")
                            
                # reset pbar
                if step < max_steps:
                    pbar = tqdm(total=log_every, desc="Training", leave=False, file=sys.stdout)
    finally:
        writer.close()
        csv_file.close()
        train_data_paths_handle.close()
        # 统计并写入不同数据数量（本 run 内参与过至少一个 step 的不同路径数）
        summary_path = temp_data_dir / "train_data_paths_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"total_unique_paths: {len(seen_data_paths)}\n")
            with open(train_data_paths_file, encoding="utf-8") as rf:
                n_lines = sum(1 for _ in rf)
            f.write(f"total_lines_in_train_data_paths_txt: {n_lines}\n")
        print(f"[TempData] 参与训练的不同数据数: {len(seen_data_paths)}，路径已追加到 {train_data_paths_file}，统计见 {summary_path}")


if __name__ == "__main__":
    main()
