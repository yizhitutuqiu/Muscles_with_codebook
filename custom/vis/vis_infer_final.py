import argparse
import json
import os
import random
import shutil
import sys
import time
import yaml
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

@dataclass(frozen=True)
class SampleRef:
    subject: str
    exercise: str
    sample_id: str
    sample_dir: Path

    @property
    def key(self) -> str:
        return f"{self.subject}/{self.exercise}/{self.sample_id}"


def _compute_official_metrics(pred: np.ndarray, gt: np.ndarray, task: str) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    
    if task == "pose2emg":
        sq_err = np.square(pred - gt)
        overall_mse = float(np.mean(sq_err))
        overall_rmse = float(np.sqrt(overall_mse))
        return {"official_global_rmse": overall_rmse}
    else:  # emg2pose
        diff = pred - gt
        l2_dist = np.sqrt(np.sum(np.square(diff), axis=-1))
        mpjpe = float(np.mean(l2_dist))
        return {"official_global_rmse": mpjpe}

def _subject_to_condval(subject: str) -> float:
    mapping = {
        "Subject0": 1.0, "Subject1": 0.9, "Subject2": 0.8, "Subject3": 0.7,
        "Subject4": 0.6, "Subject5": 0.5, "Subject6": 0.4, "Subject7": 0.3,
        "Subject8": 0.2
    }
    return mapping.get(subject, 0.1)

def _subject_to_emg_minmax(subject: str) -> tuple[np.ndarray, np.ndarray]:
    if subject == "Subject0":
        themax = np.array([226.0, 159.0, 283.0, 233.0, 406.0, 139.0, 276.0, 235.0])
        themin = np.array([7.0, 8.0, 9.0, 2.0, 9.0, 10.0, 8.0, 2.0])
    elif subject == "Subject1":
        themax = np.array([355.0, 231.0, 242.0, 128.0, 473.0, 183.0, 197.0, 98.0])
        themin = np.array([3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0])
    elif subject == "Subject2":
        themax = np.array([119.0, 178.0, 83.0, 102.0, 176.0, 106.0, 95.0, 75.0])
        themin = np.array([2.0, 2.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0])
    elif subject == "Subject3":
        themax = np.array([207.0, 97.0, 154.0, 112.0, 182.0, 122.0, 176.0, 123.0])
        themin = np.array([2.0, 3.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0])
    elif subject == "Subject4":
        themax = np.array([177.0, 125.0, 85.0, 167.0, 176.0, 110.0, 130.0, 199.0])
        themin = np.array([2.0, 3.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    elif subject == "Subject5":
        themax = np.array([213.0, 115.0, 192.0, 128.0, 207.0, 147.0, 218.0, 114.0])
        themin = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 4.0, 1.0])
    elif subject == "Subject6":
        themax = np.array([289.0, 141.0, 116.0, 179.0, 452.0, 174.0, 135.0, 177.0])
        themin = np.array([1.0, 3.0, 2.0, 7.0, 2.0, 1.0, 3.0, 4.0])
    elif subject == "Subject7":
        themax = np.array([177.0, 120.0, 175.0, 146.0, 147.0, 94.0, 243.0, 209.0])
        themin = np.array([3.0, 2.0, 2.0, 6.0, 2.0, 0.0, 2.0, 6.0])
    elif subject == "Subject8":
        themax = np.array([154.0, 53.0, 74.0, 102.0, 183.0, 59.0, 152.0, 135.0])
        themin = np.array([2.0, 3.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0])
    else:
        themax = np.array([174.0, 134.0, 125.0, 151.0, 170.0, 161.0, 119.0, 137.0])
        themin = np.array([3.0, 14.0, 4.0, 3.0, 7.0, 5.0, 8.0, 5.0])
    return themin.astype(np.float32), themax.astype(np.float32)

def _ensure_vibe_data(smpl_src_dir: Path, vibe_dst_dir: Path) -> None:
    vibe_dst_dir.mkdir(parents=True, exist_ok=True)
    has_model_files = any(p.suffix in {".pkl", ".npz"} for p in vibe_dst_dir.rglob("*") if p.is_file())
    if has_model_files: return
    if not smpl_src_dir.exists():
        raise FileNotFoundError(f"SMPL source dir not found: {smpl_src_dir}")
    shutil.copytree(smpl_src_dir, vibe_dst_dir, dirs_exist_ok=True)

def _scan_samples(dataset_root: Path, phase: str) -> list[SampleRef]:
    phase_dir = dataset_root / phase
    if not phase_dir.exists():
        raise FileNotFoundError(f"Dataset phase dir not found: {phase_dir}")
    samples = []
    for subject_dir in sorted(phase_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("Subject"): continue
        for exercise_dir in sorted(subject_dir.iterdir()):
            if not exercise_dir.is_dir(): continue
            for sample_dir in sorted(exercise_dir.iterdir()):
                if not sample_dir.is_dir(): continue
                samples.append(SampleRef(subject_dir.name, exercise_dir.name, sample_dir.name, sample_dir))
    return samples

def _require_files(sample_dir: Path) -> None:
    required = ["emgvalues.npy", "joints3d.npy", "verts.npy", "origcam.npy"]
    missing = [name for name in required if not (sample_dir / name).exists()]
    if missing: raise FileNotFoundError(f"Missing files in {sample_dir}: {missing}")

def _load_sample_arrays(sample: SampleRef) -> dict[str, np.ndarray]:
    _require_files(sample.sample_dir)
    emg = np.load(sample.sample_dir / "emgvalues.npy")
    joints3d = np.load(sample.sample_dir / "joints3d.npy")
    verts = np.load(sample.sample_dir / "verts.npy")
    origcam = np.load(sample.sample_dir / "origcam.npy")

    if emg.shape[0] == 8: emg_8_t = emg.astype(np.float32)
    elif emg.shape[1] == 8: emg_8_t = emg.T.astype(np.float32)
    else: raise ValueError(f"Expected emg to have one dim=8, got shape: {emg.shape}")

    t = joints3d.shape[0]
    if emg_8_t.shape[1] != t:
        if emg_8_t.shape[1] < t:
            joints3d, verts, origcam = joints3d[:emg_8_t.shape[1]], verts[:emg_8_t.shape[1]], origcam[:emg_8_t.shape[1]]
        else:
            emg_8_t = emg_8_t[:, :t]

    return {
        "emg_8_t": emg_8_t,
        "joints3d_t_25_3": joints3d.astype(np.float32),
        "verts_t_v_3": verts.astype(np.float32),
        "origcam_t_4": origcam.astype(np.float32),
    }

def _load_our_model(checkpoint_path: Path, device: Any) -> tuple[Any, Any]:
    from custom.tools.Mia_style_eval import _build_stage2_from_ckpt
    model, cfg, _, stage1 = _build_stage2_from_ckpt(checkpoint_path, device, stage1_override_ckpt=None)
    return model, stage1

def _infer_our_pose2emg(model, stage1, joints3d_t_25_3, condval, device) -> np.ndarray:
    import torch
    joints3d_t_25_3 = joints3d_t_25_3[:, :25, :]
    root = joints3d_t_25_3[:, 0:1, :]
    inputs_np = joints3d_t_25_3 - root
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(inputs, cond=cond)
        pred = out["pred"]
        stage1_emg = getattr(stage1, "emg", None)
        emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
        if emg_standardizer is not None:
            from custom.tools.Mia_style_eval import _emg_standardizer_stats_bt8
            mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(pred.shape[1]), device=device)
            pred = pred * std_bt8 + mean_bt8
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32).T

def _infer_our_emg2pose(model, stage1, emg_8_t, condval, device, joints3d_t_25_3) -> np.ndarray:
    import torch
    inputs_np = emg_8_t.T
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device) # B, T, 8
    
    # STANDARDIZE EMG
    stage1_emg = getattr(stage1, "emg", None)
    emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
    if emg_standardizer is not None:
        from custom.tools.Mia_style_eval import _emg_standardizer_stats_bt8
        mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(inputs.shape[1]), device=device)
        inputs = (inputs - mean_bt8) / std_bt8
        
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(inputs, cond=cond)
        pred = out["pred"]
    pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # Restore absolute coordinates using GT's Joint 8 (which is the root in Mia_style_eval)
    joints3d_t_25_3 = joints3d_t_25_3[:, :25, :]
    root_8 = joints3d_t_25_3[:, 8:9, :]
    return pred_np + root_8

def _load_model(checkpoint_path: Path, device: Any, task: str) -> Any:
    import torch
    if task == "pose2emg":
        from musclesinaction.models.modelposetoemg import TransformerEnc
    elif task == "emg2pose":
        from musclesinaction.models.modelemgtopose import TransformerEnc
    else:
        raise ValueError(f"Unknown task: {task}")

    model = TransformerEnc(
        threed="True", num_tokens=50, dim_model=128, num_classes=20, num_heads=16,
        classif=False, num_encoder_layers=8, num_decoder_layers=3, dropout_p=0.1,
        device=str(device), embedding=True, step=30,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["my_model"] if isinstance(checkpoint, dict) and "my_model" in checkpoint else checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def _render_emg_panel(emg_8_t: np.ndarray, width: int, height: int, title: str, vmax: float = 1.0) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlim(0, emg_8_t.shape[1] - 1)
    ax.set_ylim(0.0, vmax)
    for i in range(8): ax.plot(emg_8_t[i], linewidth=1.0)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"): img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    else:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def _compute_3d_bounds(gt_joints: np.ndarray, pred_joints: np.ndarray) -> dict:
    gt_j = gt_joints[:, :25, :]
    pred_j = pred_joints[:, :25, :]
    all_joints = np.concatenate([gt_j, pred_j], axis=0)
    x = all_joints[..., 0]
    y = all_joints[..., 2]
    z = all_joints[..., 1]
    
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2
    
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0
    max_range *= 1.1 # 10% padding
    
    return {
        'x_min': x_mid - max_range, 'x_max': x_mid + max_range,
        'y_min': y_mid - max_range, 'y_max': y_mid + max_range,
        'z_min': z_mid - max_range, 'z_max': z_mid + max_range,
    }

def _render_skeleton_panel(joints3d_25_3: np.ndarray, width: int, height: int, title: str, bounds: dict = None) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    
    limb_seq = [([17, 15], [238, 0, 255]), ([15, 0], [255, 0, 166]), ([0, 16], [144, 0, 255]), ([16, 18], [65, 0, 255]), ([0, 1], [255, 0, 59]), ([1, 2], [255, 77, 0]), ([2, 3], [247, 155, 0]), ([3, 4], [255, 255, 0]), ([1, 5], [158, 245, 0]), ([5, 6], [93, 255, 0]), ([6, 7], [0, 255, 0]), ([1, 8], [255, 21, 0]), ([8, 9], [6, 255, 0]), ([9, 10], [0, 255, 117]), ([10, 24], [0, 252, 255]), ([8, 12], [0, 140, 255]), ([12, 13], [0, 68, 255]), ([13, 14], [0, 14, 255]), ([24, 22], [0, 252, 255]), ([24, 24], [0, 252, 255]), ([22, 23], [0, 252, 255]), ([14, 19], [0, 14, 255]), ([14, 21], [0, 14, 255]), ([19, 20], [0, 14, 255])]
    colors = ['b','b','r','r','r','g','g','g','y','r','r','r','g','g','g','b','b','b','b','g','g','g','r','r','r']
    
    for j in range(25):
        c = colors[j]
        newc = 'blue' if c == 'b' else ('red' if c == 'r' else '#0f0f0f')
        if j in [25, 30]: newc = 'yellow'
        ax.scatter3D(joints3d_25_3[j, 0], joints3d_25_3[j, 2], joints3d_25_3[j, 1], c=newc)

    for vertices, color in limb_seq:
        ax.plot3D([joints3d_25_3[vertices[0], 0], joints3d_25_3[vertices[1], 0]],
                  [joints3d_25_3[vertices[0], 2], joints3d_25_3[vertices[1], 2]],
                  [joints3d_25_3[vertices[0], 1], joints3d_25_3[vertices[1], 1]],
                  linewidth=3, color=[c / 255.0 for c in color], alpha=0.7)

    if bounds:
        ax.set_xlim3d([bounds['x_min'], bounds['x_max']])
        ax.set_ylim3d([bounds['y_min'], bounds['y_max']])
        ax.set_zlim3d([bounds['z_min'], bounds['z_max']])
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
    else:
        minvaly, maxvaly = np.min(joints3d_25_3[:, 2]), np.max(joints3d_25_3[:, 2])
        minvalz, maxvalz = np.min(joints3d_25_3[:, 1]), np.max(joints3d_25_3[:, 1])
        ax.set_xlim3d([-1.0, 2.0])
        ax.set_zlim3d([minvalz, maxvalz])
        ax.set_ylim3d([minvaly, maxvaly])
    ax.invert_zaxis()

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(0, -90)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"): img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    else:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def _render_overlay_skeleton_panel(gt_joints_25_3: np.ndarray, pred_joints_25_3: np.ndarray, our_joints_25_3: np.ndarray, width: int, height: int, title: str, bounds: dict = None) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    
    limb_seq = [([17, 15], None), ([15, 0], None), ([0, 16], None), ([16, 18], None), ([0, 1], None), ([1, 2], None), ([2, 3], None), ([3, 4], None), ([1, 5], None), ([5, 6], None), ([6, 7], None), ([1, 8], None), ([8, 9], None), ([9, 10], None), ([10, 24], None), ([8, 12], None), ([12, 13], None), ([13, 14], None), ([24, 22], None), ([24, 24], None), ([22, 23], None), ([14, 19], None), ([14, 21], None), ([19, 20], None)]
    
    # GT (Green)
    for j in range(25):
        ax.scatter3D(gt_joints_25_3[j, 0], gt_joints_25_3[j, 2], gt_joints_25_3[j, 1], c='green', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([gt_joints_25_3[vertices[0], 0], gt_joints_25_3[vertices[1], 0]],
                  [gt_joints_25_3[vertices[0], 2], gt_joints_25_3[vertices[1], 2]],
                  [gt_joints_25_3[vertices[0], 1], gt_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='green', alpha=0.7)
                  
    # Official Pred (Red)
    for j in range(25):
        ax.scatter3D(pred_joints_25_3[j, 0], pred_joints_25_3[j, 2], pred_joints_25_3[j, 1], c='red', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([pred_joints_25_3[vertices[0], 0], pred_joints_25_3[vertices[1], 0]],
                  [pred_joints_25_3[vertices[0], 2], pred_joints_25_3[vertices[1], 2]],
                  [pred_joints_25_3[vertices[0], 1], pred_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='red', alpha=0.7)
                  
    # Our Pred (Blue)
    for j in range(25):
        ax.scatter3D(our_joints_25_3[j, 0], our_joints_25_3[j, 2], our_joints_25_3[j, 1], c='blue', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([our_joints_25_3[vertices[0], 0], our_joints_25_3[vertices[1], 0]],
                  [our_joints_25_3[vertices[0], 2], our_joints_25_3[vertices[1], 2]],
                  [our_joints_25_3[vertices[0], 1], our_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='blue', alpha=0.7)

    if bounds:
        ax.set_xlim3d([bounds['x_min'], bounds['x_max']])
        ax.set_ylim3d([bounds['y_min'], bounds['y_max']])
        ax.set_zlim3d([bounds['z_min'], bounds['z_max']])
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
    else:
        minvaly = min(np.min(gt_joints_25_3[:, 2]), np.min(pred_joints_25_3[:, 2]), np.min(our_joints_25_3[:, 2]))
        maxvaly = max(np.max(gt_joints_25_3[:, 2]), np.max(pred_joints_25_3[:, 2]), np.max(our_joints_25_3[:, 2]))
        minvalz = min(np.min(gt_joints_25_3[:, 1]), np.min(pred_joints_25_3[:, 1]), np.min(our_joints_25_3[:, 1]))
        maxvalz = max(np.max(gt_joints_25_3[:, 1]), np.max(pred_joints_25_3[:, 1]), np.max(our_joints_25_3[:, 1]))
        ax.set_xlim3d([-1.0, 2.0])
        ax.set_zlim3d([minvalz, maxvalz])
        ax.set_ylim3d([minvaly, maxvaly])

    ax.invert_zaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(0, -90)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"): img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    else:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def _resize_letterbox(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if (w, h) == (target_w, target_h): return img_bgr
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def _open_video_writer(out_path: Path, fps: int, frame_size_wh: tuple[int, int]) -> cv2.VideoWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size_wh)
    if not writer.isOpened(): raise RuntimeError(f"Failed to open video writer for {out_path}")
    return writer

def _infer_emg(model: Any, joints3d_t_25_3: np.ndarray, condval: float, device: Any) -> np.ndarray:
    import torch
    t = joints3d_t_25_3.shape[0]
    x = torch.from_numpy(joints3d_t_25_3.reshape(t, -1)).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad(): pred = model(x, cond)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

def _infer_pose(model: Any, emg_8_t: np.ndarray, condval: float, device: Any) -> np.ndarray:
    import torch
    x = torch.from_numpy(emg_8_t).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad(): pred = model(x, cond)
    pred = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return pred.reshape(pred.shape[0], 25, 3)

def _normalize_emg_for_mesh(emg_8_t: np.ndarray, subject: str) -> np.ndarray:
    themin, themax = _subject_to_emg_minmax(subject)
    return ((emg_8_t - themin.reshape(8, 1)) / themax.reshape(8, 1)).astype(np.float32)

def _pad_time_first(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] >= target_len: return arr[:target_len]
    return np.concatenate([arr, np.repeat(arr[-1:], target_len - arr.shape[0], axis=0)], axis=0)

def _pad_emg_8_t(emg_8_t: np.ndarray, target_len: int) -> np.ndarray:
    if emg_8_t.shape[1] >= target_len: return emg_8_t[:, :target_len]
    return np.concatenate([emg_8_t, np.repeat(emg_8_t[:, -1:], target_len - emg_8_t.shape[1], axis=1)], axis=1)

def _compute_color_stats(emg_mesh_8_t: np.ndarray, vmax: float = 0.5) -> dict[str, Any]:
    t = int(emg_mesh_8_t.shape[1])
    per_frame = []
    for i in range(t):
        v = emg_mesh_8_t[:, i].astype(np.float64)
        per_frame.append({
            "frame": i, "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "frac_sat": float(np.mean(v >= vmax)),
        })
    v_all = emg_mesh_8_t.astype(np.float64).reshape(-1)
    summary = {
        "vmax": float(vmax), "min": float(v_all.min()), "max": float(v_all.max()),
        "mean": float(v_all.mean()), "frac_sat": float(np.mean(v_all >= vmax)),
    }
    return {"summary": summary, "per_frame": per_frame}

def _render_sequence_cells_pose2emg(
    renderer: Any, background_bgr: np.ndarray, verts_t_v_3: np.ndarray, origcam_t_4: np.ndarray,
    gt_emg_mesh_8_t: np.ndarray, pred_emg_mesh_8_t: np.ndarray, our_emg_mesh_8_t: np.ndarray,
    gt_emg_plot_8_t: np.ndarray, pred_emg_plot_8_t: np.ndarray, our_emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, plot_vmax: float, mesh_views: str, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_emg_mesh_8_t.shape[1]
    panel_gt = cv2.cvtColor(_render_emg_panel(gt_emg_plot_8_t, plot_width, plot_height, "GT EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(_render_emg_panel(pred_emg_plot_8_t, plot_width, plot_height, "Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    for i in tqdm(range(t), desc="Rendering pose2emg frames", leave=False):
        verts, cam = verts_t_v_3[i], origcam_t_4[i]
        meshes = []
        for is_pred, emg_mesh in [(False, gt_emg_mesh_8_t), (True, pred_emg_mesh_8_t)]:
            views = []
            if mesh_views in ["front", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=True, pred=is_pred)
                views.append(v)
            if mesh_views in ["back", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=False, pred=is_pred)
                views.append(v)
            meshes.append(np.concatenate(views, axis=1))
        
        gt_mesh, pred_mesh = meshes
        frames.append(np.concatenate([np.concatenate([gt_mesh, panel_gt], axis=1), np.concatenate([pred_mesh, panel_pred], axis=1)], axis=0))
    return frames

def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool,
    align_root_to_gt: bool = False
) -> list[np.ndarray]:
    if align_root_to_gt:
        pred_joints_t_25_3 = pred_joints_t_25_3 - pred_joints_t_25_3[:, 0:1, :] + gt_joints_t_25_3[:, 0:1, :]
        our_joints_t_25_3 = our_joints_t_25_3 - our_joints_t_25_3[:, 0:1, :] + gt_joints_t_25_3[:, 0:1, :]

    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3[:, :25], pred_joints_t_25_3[:, :25], our_joints_t_25_3[:, :25]], axis=0), np.zeros_like(gt_joints_t_25_3[:, :25])) # HACK: compute_3d_bounds handles concatenated
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        our_skel = cv2.cvtColor(_render_skeleton_panel(our_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], our_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:G, Off:R, Ours:B)", bounds), cv2.COLOR_RGB2BGR)
        
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        our_row = np.concatenate([our_skel, panel_emg], axis=1)
        overlay_row = np.concatenate([overlay_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row, our_row, overlay_row], axis=0))
    return frames

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/data/litengmo/HSMR/mia_custom/custom/vis/configs/vis_infer_final.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task", "pose2emg")
    dataset_root = Path(cfg.get("dataset_root", "/data/litengmo/HSMR/mia_custom/MIADatasetOfficial"))
    phase = cfg.get("phase", "val")
    n_per_exercise = cfg.get("n_per_exercise", 3)
    seed = cfg.get("seed", 0)
    max_exercises = cfg.get("max_exercises", -1)
    filter_worst_n = cfg.get("filter_worst_n", 0)
    filter_our_best_diff_n = cfg.get("filter_our_best_diff_n", 0)
    filter_our_max_rmse_threshold = cfg.get("filter_our_max_rmse_threshold", 999.0)
    filter_diversity_exercise = cfg.get("filter_diversity_exercise", False)
    
    if task == "pose2emg":
        checkpoint_path = Path(cfg.get("pose2emg", {}).get("checkpoint", ""))
        our_checkpoint_path = Path(cfg.get("pose2emg", {}).get("our_checkpoint", ""))
    elif task == "emg2pose":
        checkpoint_path = Path(cfg.get("emg2pose", {}).get("checkpoint", ""))
        our_checkpoint_path = Path(cfg.get("emg2pose", {}).get("our_checkpoint", ""))
    else:
        raise ValueError(f"Unknown task: {task}")

    base_out_dir = cfg.get("out_dir", "/data/litengmo/HSMR/mia_custom/custom/output")
    if base_out_dir.endswith("vis_infer_final"):
        base_out_dir = os.path.dirname(base_out_dir)
    out_dir = Path(base_out_dir) / f"vis_infer_final_{task}"
    smpl_src_dir = Path(cfg.get("smpl_src", "/data/litengmo/HSMR/SMPL_models/models/smpl"))
    vibe_dst_dir = Path(cfg.get("vibe_dst", "/data/litengmo/HSMR/mia_custom/musclesinaction/vibe_data"))
    device_str = cfg.get("device", "cuda")
    fps = cfg.get("fps", 10)
    render_width = cfg.get("render_width", 360)
    render_height = cfg.get("render_height", 640)
    plot_width = cfg.get("plot_width", 420)
    plot_height = cfg.get("plot_height", 640)
    plot_emg_vmax = cfg.get("plot_emg_vmax", 300.0)
    mesh_views = cfg.get("mesh_views", "both")
    debug_color_stats = cfg.get("debug_color_stats", False)
    debug_overlay_text = cfg.get("debug_overlay_text", False)
    dry_run = cfg.get("dry_run", False)
    align_root_to_gt = cfg.get("align_root_to_gt", False)

    random.seed(seed)
    
    samples = _scan_samples(dataset_root, phase)
    samples_by_exercise = {}
    for s in samples: samples_by_exercise.setdefault(s.exercise, []).append(s)

    exercise_names = sorted(samples_by_exercise.keys())
    if max_exercises > 0: exercise_names = exercise_names[:max_exercises]

    selected_by_exercise = {}
    filter_worst_n = cfg.get("filter_worst_n", 0)
    filter_our_best_diff_n = cfg.get("filter_our_best_diff_n", 0)
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
        if filter_worst_n > 0 or filter_our_best_diff_n > 0:
            # Load ALL samples for inference to find the global extreme
            selected_by_exercise[ex] = cand
        else:
            random.shuffle(cand)
            selected_by_exercise[ex] = cand[:n_per_exercise]

    out_dir.mkdir(parents=True, exist_ok=True)
    if dry_run: return

    import torch
    device = torch.device("cpu") if device_str == "cuda" and not torch.cuda.is_available() else torch.device(device_str)
    
    if task == "pose2emg":
        _ensure_vibe_data(smpl_src_dir, vibe_dst_dir)
        from musclesinaction.vis.renderer import Renderer
        renderer = Renderer(resolution=(render_width, render_height), orig_img=True, wireframe=False)
        background = cv2.imread("backplain.png")
        if background is not None: background = _resize_letterbox(background.astype(np.uint8), render_width, render_height)
        else: background = np.zeros((render_height, render_width, 3), dtype=np.uint8)
    
    model = _load_model(checkpoint_path, device, task)
    our_model, our_stage1 = None, None
    if our_checkpoint_path.exists():
        our_model, our_stage1 = _load_our_model(our_checkpoint_path, device)

    total_inference_time = 0.0
    total_visualization_time = 0.0
    
    # Phase 1: Inference & Caching
    all_prepared = {}
    for exercise, samples_in_ex in tqdm(sorted(selected_by_exercise.items(), key=lambda x: x[0]), desc="Phase 1: Inference & Cache"):
        prepared = []
        max_t = 0
        for sample in samples_in_ex:
            arrays = _load_sample_arrays(sample)
            condval = _subject_to_condval(sample.subject)
            t = arrays["emg_8_t"].shape[1]
            max_t = max(max_t, t)
            
            # Setup temp cache directory
            temp_dir = out_dir / "temp" / exercise
            temp_dir.mkdir(parents=True, exist_ok=True)
            cache_file = temp_dir / f"{sample.sample_id}_pred.npy"
            metric_file = temp_dir / f"{sample.sample_id}_metric.json"
            
            if task == "pose2emg":
                if cache_file.exists():
                    pred_emg_8_t = np.load(cache_file)
                else:
                    t0 = time.time()
                    pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)
                    total_inference_time += time.time() - t0
                    np.save(cache_file, pred_emg_8_t)
                
                if not metric_file.exists():
                    gt_flat = arrays["emg_8_t"].T # t, 8
                    metric = _compute_official_metrics(pred_emg_8_t.T, gt_flat, task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                gt_emg_mesh = _normalize_emg_for_mesh(arrays["emg_8_t"], sample.subject)
                pred_emg_mesh = _normalize_emg_for_mesh(pred_emg_8_t, sample.subject)
                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh,
                    "raw_joints": arrays["joints3d_t_25_3"], "raw_emg": arrays["emg_8_t"]
                })
            else:
                if cache_file.exists():
                    pred_joints = np.load(cache_file)
                else:
                    t0 = time.time()
                    pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)
                    total_inference_time += time.time() - t0
                    np.save(cache_file, pred_joints)
                
                if not metric_file.exists():
                    metric = _compute_official_metrics(pred_joints, arrays["joints3d_t_25_3"][:, :25, :], task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32),
                    "raw_joints": arrays["joints3d_t_25_3"], "raw_emg": arrays["emg_8_t"]
                })

        if prepared:
            all_prepared[exercise] = (prepared, max_t)


    # Run Our Model Inference on the selected items (Phase 1.5)
    if our_checkpoint_path.exists():
        if our_model is None:
            our_model, our_stage1 = _load_our_model(our_checkpoint_path, device)
        for exercise, (prepared, max_t) in tqdm(all_prepared.items(), desc="Phase 1.5: Our Model Inference & Cache"):
            for item in prepared:
                sample = item["sample"]
                temp_dir = out_dir / "temp" / sample.exercise
                our_cache_file = temp_dir / f"{sample.sample_id}_our_pred.npy"
                our_metric_file = temp_dir / f"{sample.sample_id}_our_metric.json"
                condval = _subject_to_condval(sample.subject)
                
                if task == "pose2emg":
                    if our_cache_file.exists():
                        our_pred_emg_8_t = np.load(our_cache_file)
                    else:
                        our_pred_emg_8_t = _infer_our_pose2emg(our_model, our_stage1, item["raw_joints"], condval, device)
                        np.save(our_cache_file, our_pred_emg_8_t)
                    
                    if not our_metric_file.exists():
                        gt_flat = item["raw_emg"].T
                        metric = _compute_official_metrics(our_pred_emg_8_t.T, gt_flat, task)
                        with open(our_metric_file, 'w') as mf: json.dump(metric, mf)
                        
                    our_pred_emg_mesh = _normalize_emg_for_mesh(our_pred_emg_8_t, sample.subject)
                    item["our_pred_emg_plot"] = our_pred_emg_8_t
                    item["our_pred_emg_mesh"] = our_pred_emg_mesh
                else:
                    if our_cache_file.exists():
                        our_pred_joints = np.load(our_cache_file)
                    else:
                        our_pred_joints = _infer_our_emg2pose(our_model, our_stage1, item["raw_emg"], condval, device, item["raw_joints"])
                        np.save(our_cache_file, our_pred_joints)
                        
                    if not our_metric_file.exists():
                        metric = _compute_official_metrics(our_pred_joints, item["raw_joints"][:, :25, :], task)
                        with open(our_metric_file, 'w') as mf: json.dump(metric, mf)
                        
                    item["our_pred_joints"] = our_pred_joints

    # Global Filtering
    if filter_worst_n > 0 or filter_our_best_diff_n > 0:
        all_items_with_scores = []
        for exercise, (prepared, _) in all_prepared.items():
            temp_dir = out_dir / "temp" / exercise
            for item in prepared:
                # Get Official RMSE
                metric_file = temp_dir / f"{item['sample'].sample_id}_metric.json"
                official_rmse = 0.0
                if metric_file.exists():
                    with open(metric_file, 'r') as mf:
                        m = json.load(mf)
                        official_rmse = m.get("official_global_rmse", 0.0)
                
                # Get Our RMSE
                our_metric_file = temp_dir / f"{item['sample'].sample_id}_our_metric.json"
                our_rmse = 0.0
                if our_metric_file.exists():
                    with open(our_metric_file, 'r') as mf:
                        m = json.load(mf)
                        our_rmse = m.get("official_global_rmse", 0.0)
                
                # Apply the max threshold filter
                if filter_our_max_rmse_threshold < 999.0 and our_rmse > filter_our_max_rmse_threshold:
                    continue
                
                if filter_our_best_diff_n > 0:
                    score = official_rmse - our_rmse # Higher is better (we beat official by more)
                else:
                    score = official_rmse # Higher is worse
                    
                all_items_with_scores.append((score, item))
        
        # Sort globally descending
        all_items_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        target_n = filter_our_best_diff_n if filter_our_best_diff_n > 0 else filter_worst_n
        selected_items = []
        if filter_diversity_exercise:
            seen_exercises = set()
            for score, item in all_items_with_scores:
                ex = item["sample"].exercise
                if ex not in seen_exercises:
                    selected_items.append(item)
                    seen_exercises.add(ex)
                if len(selected_items) >= target_n:
                    break
        else:
            selected_items = [x[1] for x in all_items_with_scores[:target_n]]
        
        # Repackage into a single pseudo-exercise
        if selected_items:
            pseudo_name = "global_best_diff" if filter_our_best_diff_n > 0 else "global_worst"
            global_max_t = max(item["emg_plot"].shape[1] if task != "pose2emg" else item["gt_emg_plot"].shape[1] for item in selected_items)
            all_prepared = {pseudo_name: (selected_items, global_max_t)}
        else:
            all_prepared = {}

    # Phase 2: Visualization
    for exercise, (prepared, max_t) in tqdm(sorted(all_prepared.items(), key=lambda x: x[0]), desc="Phase 2: Visualization"):
        t1 = time.time()
        cell_streams = []
        for item in prepared:
            if task == "pose2emg":
                item["verts"] = _pad_time_first(item["verts"], max_t)
                item["cam"] = _pad_time_first(item["cam"], max_t)
                item["gt_emg_plot"] = _pad_emg_8_t(item["gt_emg_plot"], max_t)
                item["pred_emg_plot"] = _pad_emg_8_t(item["pred_emg_plot"], max_t)
                item["gt_emg_mesh"] = _pad_emg_8_t(item["gt_emg_mesh"], max_t)
                item["pred_emg_mesh"] = _pad_emg_8_t(item["pred_emg_mesh"], max_t)
                
                item["our_pred_emg_plot"] = _pad_emg_8_t(item["our_pred_emg_plot"], max_t)
                item["our_pred_emg_mesh"] = _pad_emg_8_t(item["our_pred_emg_mesh"], max_t)
                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"], item["our_pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], item["our_pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )
            else:
                item["gt_joints"] = _pad_time_first(item["gt_joints"], max_t)
                item["pred_joints"] = _pad_time_first(item["pred_joints"], max_t)
                item["emg_plot"] = _pad_emg_8_t(item["emg_plot"], max_t)
                

                item["our_pred_joints"] = _pad_time_first(item["our_pred_joints"], max_t)
                
                
                
                cell_frames = _render_sequence_cells_emg2pose(

                    item["gt_joints"], item["pred_joints"], item["our_pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )
            cell_streams.append(cell_frames)

        cell_h, cell_w = cell_streams[0][0].shape[:2]
        
        if filter_worst_n > 0 or filter_our_best_diff_n > 0:
            vid_out_dir = out_dir / "selected"
        else:
            vid_out_dir = out_dir / exercise
        
        writer = _open_video_writer(vid_out_dir / f"grid_{task}_n{len(prepared)}.mp4", fps, (cell_w * len(cell_streams), cell_h))
        try:
            for i in range(max_t):
                writer.write(np.concatenate([cell_streams[c][i] for c in range(len(cell_streams))], axis=1))
        finally:
            writer.release()
        total_visualization_time += time.time() - t1

    print(f"\n--- Time Statistics ---")
    print(f"Total Inference Time: {total_inference_time:.2f} s")
    print(f"Total Visualization Time: {total_visualization_time:.2f} s")
    if total_inference_time > 0:
        print(f"Visualization takes {total_visualization_time / total_inference_time:.1f}x longer than inference.")

if __name__ == "__main__":
    main()
