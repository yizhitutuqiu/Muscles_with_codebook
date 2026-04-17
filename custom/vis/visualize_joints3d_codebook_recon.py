from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml

try:
    import imageio
except ImportError:
    imageio = None  # type: ignore

# File path:
#   .../golf_third_party/musclesinaction/custom/vis/visualize_joints3d_codebook_recon.py
_MIA_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.models.frame_codebook import FrameCodebookConfig, FrameCodebookModel, ModalityConfig  # noqa: E402
from custom.models.temporal import TemporalConfig  # noqa: E402
from custom.models.vq_ema import VQEMAConfig  # noqa: E402
from custom.utils.mia_filelist import build_mia_train_filelist  # noqa: E402
from custom.utils.online_standardize import OnlineStandardizeConfig  # noqa: E402
from custom.utils.path_utils import get_musclesinaction_repo_root  # noqa: E402


BODY25_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
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


def _build_optional_modality_cfg(name: str, d: Dict[str, Any] | None) -> ModalityConfig | None:
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


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = [p for p in ckpt_dir.glob("*.pt") if p.is_file()]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


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


def _maybe_build_filelist(
    *,
    mia_root: Path,
    split: str,
    out_txt: Path,
    max_samples: int | None = None,
) -> Path:
    if out_txt.exists() and out_txt.stat().st_size > 0:
        return out_txt
    res = build_mia_train_filelist(
        mia_repo_root=mia_root,
        split=split,
        out_txt=out_txt,
        max_samples=max_samples,
        require_files=["joints3d.npy"],
    )
    print(f"[FileList] wrote {res.num_samples} samples -> {res.filelist_path}")
    return res.filelist_path


def _denormalize_from_modality(mod, x_n: torch.Tensor) -> torch.Tensor:
    stdzr = getattr(mod, "standardizer", None)
    if stdzr is None:
        return x_n
    std = torch.sqrt(torch.clamp_min(stdzr.var, stdzr.cfg.clip_std_min**2) + float(stdzr.cfg.eps))
    return x_n * std + stdzr.mean


@torch.no_grad()
def _reconstruct_joints3d_frame(model: FrameCodebookModel, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    x: (1, 75) frame mode or (1, clip_len*75) clip mode — one sample (one frame or one clip).
    Returns:
      x_hat_denorm: same shape as x (1, 75) or (1, clip_len*75)
      info: indices/perplexity/usage summary
    """
    mod = model.joints3d
    x = x.float()
    x_n = mod.normalize(x, update=False)
    b = int(x_n.shape[0])
    token_count = int(mod.cfg.token_count)
    code_dim = int(mod.cfg.code_dim)

    z_e = mod.encoder(x_n).view(b, token_count, code_dim)
    if getattr(model.cfg, "encoder_decoder_only", False):
        z_q = z_e.view(b, token_count * code_dim)
        info = {
            "vq_loss": 0.0,
            "perplexity": 0.0,
            "codes_unique": 0,
            "token_count": token_count,
            "indices": [0] * token_count,
            "usage_mean": 0.0,
            "usage_min": 0.0,
            "usage_max": 0.0,
        }
    else:
        z_q_tokens, indices, vq_loss, perplexity, usage = model.vq(z_e.reshape(b * token_count, code_dim))
        z_q = z_q_tokens.view(b, token_count * code_dim)
        info = {
            "vq_loss": float(vq_loss.item()),
            "perplexity": float(perplexity.item()),
            "codes_unique": int(torch.unique(indices).numel()),
            "token_count": token_count,
            "indices": indices.view(b, token_count)[0].detach().cpu().tolist(),
            "usage_mean": float(usage.mean().item()),
            "usage_min": float(usage.min().item()),
            "usage_max": float(usage.max().item()),
        }
    x_hat_n = mod.decoder(z_q)
    x_hat = _denormalize_from_modality(mod, x_hat_n)
    return x_hat, info


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


def _save_visualization(
    *,
    out_png: Path,
    orig: np.ndarray,
    recon: np.ndarray,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _draw_skeleton(ax1, orig, "Original 3D Joints", "#1f77b4", "#1f77b4")
    _draw_skeleton(ax2, recon, "Codebook Reconstruction", "#d62728", "#d62728")

    _set_equal_3d_axes(ax1, orig, recon)
    _set_equal_3d_axes(ax2, orig, recon)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fit_weak_perspective(pts3d: np.ndarray, pts2d: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit a weak-perspective camera:
      u = sx * x + tx
      v = sy * y + ty

    We intentionally fit x/y scales independently. In practice, the dataset 2D image coordinates
    often use y-down while 3D joints use y-up, so forcing a shared scale can visually flip the
    reconstruction upside down.
    """
    pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
    pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 2)
    if pts3d.shape[0] != pts2d.shape[0]:
        raise ValueError(f"Point count mismatch: {pts3d.shape[0]} vs {pts2d.shape[0]}")

    ax = np.stack([pts3d[:, 0], np.ones(pts3d.shape[0], dtype=np.float64)], axis=1)
    bx = pts2d[:, 0]
    sol_x, _, _, _ = np.linalg.lstsq(ax, bx, rcond=None)
    sx, tx = [float(v) for v in sol_x]

    ay = np.stack([pts3d[:, 1], np.ones(pts3d.shape[0], dtype=np.float64)], axis=1)
    by = pts2d[:, 1]
    sol_y, _, _, _ = np.linalg.lstsq(ay, by, rcond=None)
    sy, ty = [float(v) for v in sol_y]
    return sx, sy, tx, ty


def _project_weak_perspective(pts3d: np.ndarray, camera: Tuple[float, float, float, float]) -> np.ndarray:
    sx, sy, tx, ty = camera
    pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((pts3d.shape[0], 2), dtype=np.float64)
    out[:, 0] = sx * pts3d[:, 0] + tx
    out[:, 1] = sy * pts3d[:, 1] + ty
    return out.astype(np.float32)


def _root_center_joints3d_np(j3d: np.ndarray, *, root_index: int) -> np.ndarray:
    """
    j3d: (T,25,3) or (25,3)
    """
    j3d = np.asarray(j3d, dtype=np.float32)
    if j3d.ndim == 2:
        if j3d.shape != (25, 3):
            raise ValueError(f"Expected (25,3), got {j3d.shape}")
        root = j3d[root_index : root_index + 1]
        return j3d - root
    if j3d.ndim == 3:
        if j3d.shape[1:] != (25, 3):
            raise ValueError(f"Expected (T,25,3), got {j3d.shape}")
        root = j3d[:, root_index : root_index + 1, :]
        return j3d - root
    raise ValueError(f"Unsupported j3d shape: {j3d.shape}")


def _draw_skeleton_2d(ax, pts: np.ndarray, title: str, color_pts: str, color_edges: str, alpha: float = 1.0) -> None:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ax.scatter(pts[:, 0], pts[:, 1], c=color_pts, s=22, alpha=alpha)
    for i, j in BODY25_EDGES:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            color=color_edges,
            linewidth=1.5,
            alpha=alpha,
        )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")


def _set_equal_2d_axes(ax, pts_a: np.ndarray, pts_b: np.ndarray) -> None:
    pts = np.concatenate([pts_a, pts_b], axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    # Image coordinates are y-down. Set reversed limits explicitly instead of relying on
    # `invert_yaxis()`, because later `set_ylim()` would override the inversion.
    ax.set_ylim(center[1] + radius, center[1] - radius)


def _save_visualization_2d(
    *,
    out_png: Path,
    orig_2d: np.ndarray,
    recon_2d: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    _draw_skeleton_2d(ax1, orig_2d, "Original 2D Joints", "#1f77b4", "#1f77b4")
    _draw_skeleton_2d(ax2, recon_2d, "Reconstructed 2D Projection", "#d62728", "#d62728")
    _draw_skeleton_2d(ax3, orig_2d, "Overlay", "#1f77b4", "#1f77b4", alpha=0.9)
    _draw_skeleton_2d(ax3, recon_2d, "Overlay", "#d62728", "#d62728", alpha=0.7)

    _set_equal_2d_axes(ax1, orig_2d, recon_2d)
    _set_equal_2d_axes(ax2, orig_2d, recon_2d)
    _set_equal_2d_axes(ax3, orig_2d, recon_2d)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _global_2d_limits(orig_2d_list: np.ndarray, recon_2d_list: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute axis limits from all frames so overlay panels/video share the same scale."""
    all_pts = np.concatenate([orig_2d_list.reshape(-1, 2), recon_2d_list.reshape(-1, 2)], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    radius *= 1.15  # 15% padding so keypoints are not cramped at edges
    return (
        center[0] - radius,
        center[0] + radius,
        center[1] + radius,
        center[1] - radius,
    )  # xmin, xmax, ymin, ymax (y-down for image coords)


def _save_clip_30_overlay_png(
    *,
    out_png: Path,
    orig_2d_list: np.ndarray,
    recon_2d_list: np.ndarray,
    title: str,
) -> None:
    """
    Save one big image: 3 rows x 10 cols, each cell = overlay (original + reconstructed).
    orig_2d_list / recon_2d_list: (30, 25, 2). Larger subplot size so keypoints are not cramped.
    """
    n_frames = int(orig_2d_list.shape[0])
    nrows, ncols = 3, 10
    if n_frames != nrows * ncols:
        raise ValueError(f"Expected 30 frames, got {n_frames}")
    xmin, xmax, ymin, ymax = _global_2d_limits(orig_2d_list, recon_2d_list)

    # Each subplot ~2.8 inch so skeleton has room; total fig (28, 8.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.8))
    axes = axes.flatten()
    for t, ax in enumerate(axes):
        _draw_skeleton_2d(ax, orig_2d_list[t], "", "#1f77b4", "#1f77b4", alpha=0.9)
        _draw_skeleton_2d(ax, recon_2d_list[t], "", "#d62728", "#d62728", alpha=0.7)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t={t}", fontsize=9)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", linewidth=3, label="Original"),
            Line2D([0], [0], color="#d62728", linewidth=3, label="Reconstructed"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _resize_frame_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        import cv2
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        if img.ndim == 2:
            pil = Image.fromarray(img).convert("L")
        else:
            pil = Image.fromarray(img).convert("RGB")
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
        arr = np.array(pil)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr


def _save_clip_overlay_mp4(
    *,
    out_mp4: Path,
    orig_2d_list: np.ndarray,
    recon_2d_list: np.ndarray,
    fps: int = 15,
    width: int = 640,
    height: int = 480,
    dpi: int = 100,
) -> None:
    """
    Save overlay as MP4: one frame per timestep, 2D overlay only. Uses imageio get_writer + append_data
    so the file is actually written and closed (mimsave can produce 0-byte files if backend fails).
    """
    if imageio is None:
        raise RuntimeError("Clip-mode MP4 output requires imageio. Install with: pip install imageio[ffmpeg]")
    n_frames = int(orig_2d_list.shape[0])
    xmin, xmax, ymin, ymax = _global_2d_limits(orig_2d_list, recon_2d_list)

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(out_mp4.resolve())

    # get_writer + append_data + close so data is flushed (mimsave can yield 0-byte file if backend fails)
    try:
        writer = imageio.get_writer(path_str, format="FFMPEG", fps=fps, codec="libx264")
    except TypeError:
        writer = imageio.get_writer(path_str, fps=fps)

    try:
        for t in range(n_frames):
            fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi))
            _draw_skeleton_2d(ax, orig_2d_list[t], "Overlay", "#1f77b4", "#1f77b4", alpha=0.9)
            _draw_skeleton_2d(ax, recon_2d_list[t], "Overlay", "#d62728", "#d62728", alpha=0.7)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(
                handles=[
                    Line2D([0], [0], color="#1f77b4", linewidth=2, label="Original"),
                    Line2D([0], [0], color="#d62728", linewidth=2, label="Reconstructed"),
                ],
                loc="upper right",
            )
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img = imageio.imread(buf)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[..., :3].copy()
            img = _resize_frame_to(img, height, width)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            writer.append_data(img)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="custom/configs/frame_codebook_mia_train.yaml",
        help="Training config yaml. If checkpoint contains config, that one is preferred.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, use latest checkpoint under custom/checkpoints/frame_codebook.",
    )
    parser.add_argument("--split", type=str, default="test", help="train / val / test. If test missing, fallback to val.")
    parser.add_argument("--sample_idx", type=int, default=None, help="Optional sample index in the chosen split.")
    parser.add_argument("--frame_idx", type=int, default=None, help="Optional frame index in [0,29].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--vis_3d",
        action="store_true",
        help="If set, save the old 3D side-by-side visualization. By default, save 2D projection visualization.",
    )
    parser.add_argument(
        "--eval_in_train_mode",
        action="store_true",
        help="If set, run model in model.train() mode (e.g. to compare with training-time behavior / OnlineStandardizer).",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    split = _resolve_split(args.split, mia_root)
    default_ckpt_dir = mia_root / "custom" / "checkpoints" / "frame_codebook"
    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _find_latest_checkpoint(default_ckpt_dir)
    payload = torch.load(ckpt_path, map_location="cpu")

    cfg: Dict[str, Any]
    if isinstance(payload, dict) and "config" in payload:
        cfg = payload["config"]
    else:
        cfg = _load_yaml(Path(args.config).expanduser().resolve())

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = _build_model_from_cfg(cfg, device)
    model.load_state_dict(payload["model_state"], strict=True)
    if args.eval_in_train_mode:
        model.train()
        print("[Vis] eval_in_train_mode=True: 使用 model.train() 跑可视化")
    else:
        model.eval()

    data_cfg = cfg.get("data", {}) or {}
    pack_cfg = (data_cfg.get("pack", {}) or {}) if isinstance(data_cfg, dict) else {}
    pack_mode = str(pack_cfg.get("mode", "frame")).strip().lower()
    clip_len = int(pack_cfg.get("clip_len", data_cfg.get("step", 30)))
    joints3d_root_center = bool(data_cfg.get("joints3d_root_center", True))
    joints3d_root_index = int(data_cfg.get("joints3d_root_index", 8))

    filelist_name = f"miaofficial_{split}_vis.txt"
    filelist_path = mia_root / "custom" / "tools" / "datasetsplits" / filelist_name
    _maybe_build_filelist(mia_root=mia_root, split=split, out_txt=filelist_path)

    from musclesinaction.dataloader.data import MyMuscleDataset  # type: ignore

    dset = MyMuscleDataset(
        str(filelist_path),
        _NullLogger(),
        split,
        percent=1.0,
        step=30,
        std="False",
        cond="True",
        transform=None,
    )

    sample_idx = int(args.sample_idx) if args.sample_idx is not None else random.randrange(len(dset))
    sample = dset[sample_idx]
    joints3d = torch.as_tensor(sample["3dskeleton"], dtype=torch.float32)  # (T,25,3), T typically 30
    joints2d = torch.as_tensor(sample["2dskeleton"], dtype=torch.float32)  # (T,25,2)
    T = int(joints3d.shape[0])
    if T < clip_len:
        raise ValueError(f"Sample has T={T} < clip_len={clip_len}; need at least {clip_len} frames for clip mode.")
    frame_idx = int(args.frame_idx) if args.frame_idx is not None else random.randrange(min(T, clip_len))
    if frame_idx < 0 or frame_idx >= clip_len:
        raise ValueError(f"frame_idx must be in [0,{clip_len - 1}], got {frame_idx}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_rel = str(sample.get("filepath", "unknown_sample"))
    sample_name = sample_rel.replace("/", "__")
    out_dir = mia_root / "custom" / "output" / "joints3d_codebook_recon" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if pack_mode in ("clip", "sequence", "seq"):
        # Clip mode: one forward over full clip; output 30-panel overlay PNG + overlay MP4 (15 fps).
        # IMPORTANT: match training preprocessing (root-centering) before feeding to the codebook.
        j3d_np = joints3d[:clip_len].detach().cpu().numpy()
        if joints3d_root_center:
            j3d_np = _root_center_joints3d_np(j3d_np, root_index=joints3d_root_index)
        clip_input = torch.from_numpy(j3d_np).reshape(1, clip_len * 75).to(device)
        clip_recon, info = _reconstruct_joints3d_frame(model, clip_input)
        clip_recon = clip_recon.view(clip_len, 25, 3).detach().cpu().numpy()
        orig_3d_np = j3d_np   # (30, 25, 3) in the same coordinate system as training/model
        orig_2d_list = joints2d[:clip_len].detach().cpu().numpy()  # (30, 25, 2)
        recon_2d_list = np.zeros_like(orig_2d_list)
        for t in range(clip_len):
            cam_t = _fit_weak_perspective(orig_3d_np[t], orig_2d_list[t])
            recon_2d_list[t] = _project_weak_perspective(clip_recon[t], cam_t)
        out_png = out_dir / f"{sample_name}_clip30_overlay.png"
        out_mp4 = out_dir / f"{sample_name}_clip30_overlay.mp4"
        title = f"Clip overlay split={split} sample_idx={sample_idx} (blue=Original, red=Reconstructed)"
        _save_clip_30_overlay_png(
            out_png=out_png,
            orig_2d_list=orig_2d_list,
            recon_2d_list=recon_2d_list,
            title=title,
        )
        try:
            _save_clip_overlay_mp4(
                out_mp4=out_mp4,
                orig_2d_list=orig_2d_list,
                recon_2d_list=recon_2d_list,
                fps=15,
            )
        except Exception as e:
            print("[Warn] Clip MP4 write failed:", e)
        # Per-frame RMSE for meta (use frame 0 as representative)
        orig_np = orig_3d_np[0]
        recon_np = clip_recon[0]
        orig_2d_np = orig_2d_list[0]
        recon_2d_np = recon_2d_list[0]
        mse = float(np.mean((orig_np - recon_np) ** 2))
        rmse = float(math.sqrt(mse))
        mse_2d = float(np.mean((orig_2d_np - recon_2d_np) ** 2))
        rmse_2d = float(math.sqrt(mse_2d))
        camera = _fit_weak_perspective(orig_np, orig_2d_np)
        vis_mode = "clip"
        out_json = out_dir / f"{sample_name}_clip30.json"
        out_npy = out_dir / f"{sample_name}_clip30.npz"
        meta = {
            "checkpoint": str(ckpt_path),
            "split": split,
            "sample_idx": sample_idx,
            "sample_filepath": sample_rel,
            "pack_mode": pack_mode,
            "clip_len": clip_len,
            "joints3d_root_center": joints3d_root_center,
            "joints3d_root_index": joints3d_root_index,
            "vis_mode": vis_mode,
            "rmse_3d_frame0": rmse,
            "mse_3d_frame0": mse,
            "rmse_2d_frame0": rmse_2d,
            "mse_2d_frame0": mse_2d,
            "vq_info": info,
            "output_png": str(out_png),
            "output_mp4": str(out_mp4),
            "output_json": str(out_json),
            "output_npz": str(out_npy),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.savez_compressed(
            out_npy,
            original_3d_clip=orig_3d_np,
            reconstructed_3d_clip=clip_recon,
            original_2d_clip=orig_2d_list,
            reconstructed_2d_clip=recon_2d_list,
        )
        # 额外输出：每个 token 命中的码本条目编号（token 0 对应第一帧时间位置）
        out_codebook_indices = out_dir / f"{sample_name}_clip30_codebook_indices.json"
        codebook_indices_meta = {
            "description": "每个 token 查询码本时命中的条目编号；token_index 0 对应第一帧时间位置",
            "token_count": info["token_count"],
            "joints3d_codebook_indices": info["indices"],
            "first_frame_token_index": 0,
            "first_frame_codebook_entry": info["indices"][0],
            "codes_unique": info["codes_unique"],
        }
        with open(out_codebook_indices, "w", encoding="utf-8") as f:
            json.dump(codebook_indices_meta, f, ensure_ascii=False, indent=2)
        print("[Done] saved_codebook_indices =", out_codebook_indices)
        print("[Done] checkpoint =", ckpt_path)
        print("[Done] split =", split)
        print("[Done] pack_mode =", pack_mode)
        print("[Done] eval_in_train_mode =", getattr(args, "eval_in_train_mode", False))
        print("[Done] sample_idx =", sample_idx)
        print("[Done] vis_mode =", vis_mode)
        print("[Done] saved_png (30 panels) =", out_png)
        print("[Done] saved_mp4 (15 fps) =", out_mp4)
        print("[Done] saved_json =", out_json)
        print("[Done] saved_npz =", out_npy)
        return
    # Frame mode below.
    frame_idx = int(args.frame_idx) if args.frame_idx is not None else random.randrange(min(T, clip_len))
    if frame_idx < 0 or frame_idx >= clip_len:
        raise ValueError(f"frame_idx must be in [0,{clip_len - 1}], got {frame_idx}")

    frame_np = joints3d[frame_idx].detach().cpu().numpy()
    if joints3d_root_center:
        frame_np = _root_center_joints3d_np(frame_np, root_index=joints3d_root_index)
    frame_orig = torch.from_numpy(frame_np).reshape(1, 75).to(device)
    frame_recon, info = _reconstruct_joints3d_frame(model, frame_orig)
    orig_np = frame_orig.detach().cpu().reshape(25, 3).numpy()
    recon_np = frame_recon.detach().cpu().reshape(25, 3).numpy()
    orig_2d_np = joints2d[frame_idx].detach().cpu().reshape(25, 2).numpy()

    mse = float(np.mean((orig_np - recon_np) ** 2))
    rmse = float(math.sqrt(mse))
    camera = _fit_weak_perspective(orig_np, orig_2d_np)
    recon_2d_np = _project_weak_perspective(recon_np, camera)
    mse_2d = float(np.mean((orig_2d_np - recon_2d_np) ** 2))
    rmse_2d = float(math.sqrt(mse_2d))

    out_png = out_dir / f"{sample_name}_frame{frame_idx:02d}.png"
    out_json = out_dir / f"{sample_name}_frame{frame_idx:02d}.json"
    out_npy = out_dir / f"{sample_name}_frame{frame_idx:02d}.npz"

    if bool(args.vis_3d):
        title = f"3D split={split} sample_idx={sample_idx} frame={frame_idx} rmse3d={rmse:.6f}"
        _save_visualization(out_png=out_png, orig=orig_np, recon=recon_np, title=title)
        vis_mode = "3d"
    else:
        title = f"2D split={split} sample_idx={sample_idx} frame={frame_idx} rmse2d={rmse_2d:.6f}"
        _save_visualization_2d(out_png=out_png, orig_2d=orig_2d_np, recon_2d=recon_2d_np, title=title)
        vis_mode = "2d"

    meta = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "sample_idx": sample_idx,
        "sample_filepath": sample_rel,
        "frame_idx": frame_idx,
        "pack_mode": pack_mode,
        "clip_len": clip_len,
        "joints3d_root_center": joints3d_root_center,
        "joints3d_root_index": joints3d_root_index,
        "vis_mode": vis_mode,
        "rmse_3d": rmse,
        "mse_3d": mse,
        "rmse_2d": rmse_2d,
        "mse_2d": mse_2d,
        "weak_perspective_camera": {
            "sx": float(camera[0]),
            "sy": float(camera[1]),
            "tx": float(camera[2]),
            "ty": float(camera[3]),
        },
        "vq_info": info,
        "output_png": str(out_png),
        "output_npz": str(out_npy),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        out_npy,
        original_3d=orig_np,
        reconstructed_3d=recon_np,
        original_2d=orig_2d_np,
        reconstructed_2d=recon_2d_np,
    )
    out_codebook_indices = out_dir / f"{sample_name}_frame{frame_idx:02d}_codebook_indices.json"
    codebook_indices_meta = {
        "description": "该帧每个 token 查询码本时命中的条目编号",
        "frame_idx": frame_idx,
        "token_count": info["token_count"],
        "joints3d_codebook_indices": info["indices"],
        "codes_unique": info["codes_unique"],
    }
    with open(out_codebook_indices, "w", encoding="utf-8") as f:
        json.dump(codebook_indices_meta, f, ensure_ascii=False, indent=2)
    print("[Done] saved_codebook_indices =", out_codebook_indices)

    print("[Done] checkpoint =", ckpt_path)
    print("[Done] split =", split)
    print("[Done] pack_mode =", pack_mode)
    print("[Done] eval_in_train_mode =", getattr(args, "eval_in_train_mode", False))
    print("[Done] sample_idx =", sample_idx)
    print("[Done] frame_idx =", frame_idx)
    print("[Done] vis_mode =", vis_mode)
    print("[Done] rmse_3d =", rmse)
    print("[Done] rmse_2d =", rmse_2d)
    print("[Done] saved_png =", out_png)
    print("[Done] saved_json =", out_json)
    print("[Done] saved_npz =", out_npy)


if __name__ == "__main__":
    main()
