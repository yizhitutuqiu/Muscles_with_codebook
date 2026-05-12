import os

code = """import argparse
import json
import os
import random
import shutil
import sys
import yaml
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

def _render_skeleton_panel(joints3d_25_3: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
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

    minvaly, maxvaly = np.min(joints3d_25_3[:, 2]), np.max(joints3d_25_3[:, 2])
    minvalz, maxvalz = np.min(joints3d_25_3[:, 1]), np.max(joints3d_25_3[:, 1])

    ax.set_xlim3d([-1.0, 2.0])
    ax.set_zlim3d([minvalz, maxvalz])
    ax.set_ylim3d([minvaly, maxvaly])
    ax.invert_zaxis()

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(0, 180)

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
    gt_emg_mesh_8_t: np.ndarray, pred_emg_mesh_8_t: np.ndarray, gt_emg_plot_8_t: np.ndarray, pred_emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, plot_vmax: float, mesh_views: str, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_emg_mesh_8_t.shape[1]
    panel_gt = cv2.cvtColor(_render_emg_panel(gt_emg_plot_8_t, plot_width, plot_height, "GT EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(_render_emg_panel(pred_emg_plot_8_t, plot_width, plot_height, "Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    for i in range(t):
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
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    for i in range(t):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i], render_width, render_height, "GT 3D Pose"), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i], render_width, render_height, "Pred 3D Pose"), cv2.COLOR_RGB2BGR)
        
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row], axis=0))
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
    
    if task == "pose2emg":
        checkpoint_path = Path(cfg.get("pose2emg", {}).get("checkpoint", ""))
    elif task == "emg2pose":
        checkpoint_path = Path(cfg.get("emg2pose", {}).get("checkpoint", ""))
    else:
        raise ValueError(f"Unknown task: {task}")

    out_dir = Path(cfg.get("out_dir", "/data/litengmo/HSMR/mia_custom/custom/output/vis_infer_final"))
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

    random.seed(seed)
    
    samples = _scan_samples(dataset_root, phase)
    samples_by_exercise = {}
    for s in samples: samples_by_exercise.setdefault(s.exercise, []).append(s)

    exercise_names = sorted(samples_by_exercise.keys())
    if max_exercises > 0: exercise_names = exercise_names[:max_exercises]

    selected_by_exercise = {}
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
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

    for exercise, samples_in_ex in sorted(selected_by_exercise.items(), key=lambda x: x[0]):
        prepared = []
        max_t = 0
        for sample in samples_in_ex:
            arrays = _load_sample_arrays(sample)
            condval = _subject_to_condval(sample.subject)
            t = arrays["emg_8_t"].shape[1]
            max_t = max(max_t, t)
            
            if task == "pose2emg":
                pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)
                gt_emg_mesh = _normalize_emg_for_mesh(arrays["emg_8_t"], sample.subject)
                pred_emg_mesh = _normalize_emg_for_mesh(pred_emg_8_t, sample.subject)
                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh,
                })
            else:
                pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)
                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32)
                })

        if not prepared: continue

        cell_streams = []
        for item in prepared:
            if task == "pose2emg":
                item["verts"] = _pad_time_first(item["verts"], max_t)
                item["cam"] = _pad_time_first(item["cam"], max_t)
                item["gt_emg_plot"] = _pad_emg_8_t(item["gt_emg_plot"], max_t)
                item["pred_emg_plot"] = _pad_emg_8_t(item["pred_emg_plot"], max_t)
                item["gt_emg_mesh"] = _pad_emg_8_t(item["gt_emg_mesh"], max_t)
                item["pred_emg_mesh"] = _pad_emg_8_t(item["pred_emg_mesh"], max_t)
                
                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )
            else:
                item["gt_joints"] = _pad_time_first(item["gt_joints"], max_t)
                item["pred_joints"] = _pad_time_first(item["pred_joints"], max_t)
                item["emg_plot"] = _pad_emg_8_t(item["emg_plot"], max_t)
                
                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )
            cell_streams.append(cell_frames)

        cell_h, cell_w = cell_streams[0][0].shape[:2]
        writer = _open_video_writer(out_dir / exercise / f"grid_{task}_n{len(prepared)}.mp4", fps, (cell_w * len(cell_streams), cell_h))
        try:
            for i in range(max_t):
                writer.write(np.concatenate([cell_streams[c][i] for c in range(len(cell_streams))], axis=1))
        finally:
            writer.release()

if __name__ == "__main__":
    main()
"""

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(code)

