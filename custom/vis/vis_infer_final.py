import argparse
import json
import os
import random
import shutil
import sys
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
    if subject == "Subject0":
        return 1.0
    if subject == "Subject1":
        return 0.9
    if subject == "Subject2":
        return 0.8
    if subject == "Subject3":
        return 0.7
    if subject == "Subject4":
        return 0.6
    if subject == "Subject5":
        return 0.5
    if subject == "Subject6":
        return 0.4
    if subject == "Subject7":
        return 0.3
    if subject == "Subject8":
        return 0.2
    return 0.1


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
    if has_model_files:
        return
    if not smpl_src_dir.exists():
        raise FileNotFoundError(f"SMPL source dir not found: {smpl_src_dir}")
    shutil.copytree(smpl_src_dir, vibe_dst_dir, dirs_exist_ok=True)


def _scan_samples(dataset_root: Path, phase: str) -> list[SampleRef]:
    phase_dir = dataset_root / phase
    if not phase_dir.exists():
        raise FileNotFoundError(f"Dataset phase dir not found: {phase_dir}")

    samples: list[SampleRef] = []
    for subject_dir in sorted(phase_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("Subject"):
            continue
        for exercise_dir in sorted(subject_dir.iterdir()):
            if not exercise_dir.is_dir():
                continue
            for sample_dir in sorted(exercise_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                sample_id = sample_dir.name
                samples.append(
                    SampleRef(
                        subject=subject_dir.name,
                        exercise=exercise_dir.name,
                        sample_id=sample_id,
                        sample_dir=sample_dir,
                    )
                )
    return samples


def _require_files(sample_dir: Path) -> None:
    required = [
        "emgvalues.npy",
        "joints3d.npy",
        "verts.npy",
        "origcam.npy",
    ]
    missing = [name for name in required if not (sample_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {sample_dir}: {missing}")


def _load_sample_arrays(sample: SampleRef) -> dict[str, np.ndarray]:
    _require_files(sample.sample_dir)
    emg = np.load(sample.sample_dir / "emgvalues.npy")
    joints3d = np.load(sample.sample_dir / "joints3d.npy")
    verts = np.load(sample.sample_dir / "verts.npy")
    origcam = np.load(sample.sample_dir / "origcam.npy")

    if emg.ndim != 2:
        raise ValueError(f"Unexpected emg shape: {emg.shape} in {sample.sample_dir}")
    if joints3d.ndim != 3 or joints3d.shape[-1] != 3:
        raise ValueError(f"Unexpected joints3d shape: {joints3d.shape} in {sample.sample_dir}")
    if origcam.ndim != 2 or origcam.shape[-1] != 4:
        raise ValueError(f"Unexpected origcam shape: {origcam.shape} in {sample.sample_dir}")

    if emg.shape[0] == 8:
        emg_8_t = emg.astype(np.float32)
    elif emg.shape[1] == 8:
        emg_8_t = emg.T.astype(np.float32)
    else:
        raise ValueError(f"Expected emg to have one dim=8, got shape: {emg.shape}")

    t = joints3d.shape[0]
    if emg_8_t.shape[1] != t:
        if emg_8_t.shape[1] < t:
            joints3d = joints3d[: emg_8_t.shape[1]]
            verts = verts[: emg_8_t.shape[1]]
            origcam = origcam[: emg_8_t.shape[1]]
        else:
            emg_8_t = emg_8_t[:, :t]

    return {
        "emg_8_t": emg_8_t,
        "joints3d_t_25_3": joints3d.astype(np.float32),
        "verts_t_v_3": verts.astype(np.float32),
        "origcam_t_4": origcam.astype(np.float32),
    }


def _load_model(checkpoint_path: Path, device: Any) -> Any:
    import torch
    from musclesinaction.models.modelposetoemg import TransformerEnc

    model = TransformerEnc(
        threed="True",
        num_tokens=50,
        dim_model=128,
        num_classes=20,
        num_heads=16,
        classif=False,
        num_encoder_layers=8,
        num_decoder_layers=3,
        dropout_p=0.1,
        device=str(device),
        embedding=True,
        step=30,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "my_model" in checkpoint:
        state_dict = checkpoint["my_model"]
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _render_emg_panel(
    emg_8_t: np.ndarray,
    width: int,
    height: int,
    title: str,
    vmax: float = 1.0,
) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlim(0, emg_8_t.shape[1] - 1)
    ax.set_ylim(0.0, vmax)
    for i in range(8):
        ax.plot(emg_8_t[i], linewidth=1.0)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = rgba[..., :3].copy()
    elif hasattr(fig.canvas, "tostring_rgb"):
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    elif hasattr(fig.canvas, "tostring_argb"):
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        argb = argb.reshape((h, w, 4))
        img = argb[..., 1:4].copy()
    else:
        raise RuntimeError("Unsupported matplotlib canvas backend")
    plt.close(fig)
    return img


def _write_video_mp4(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                raise ValueError(f"Frame size mismatch: expected {(h, w)}, got {frame.shape[:2]}")
            writer.write(frame)
    finally:
        writer.release()


def _open_video_writer(out_path: Path, fps: int, frame_size_wh: tuple[int, int]) -> cv2.VideoWriter:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = frame_size_wh
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")
    return writer


def _infer_emg(
    model: Any,
    joints3d_t_25_3: np.ndarray,
    condval: float,
    device: Any,
) -> np.ndarray:
    import torch

    t = joints3d_t_25_3.shape[0]
    x = torch.from_numpy(joints3d_t_25_3.reshape(t, -1)).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x, cond)
    pred = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if pred.shape[0] != 8:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")
    return pred


def _normalize_emg(emg_8_t: np.ndarray, subject: str) -> np.ndarray:
    themin, themax = _subject_to_emg_minmax(subject)
    emg_norm = (emg_8_t - themin.reshape(8, 1)) / themax.reshape(8, 1)
    emg_norm = np.clip(emg_norm, 0.0, 1.0)
    return emg_norm.astype(np.float32)


def _render_sequence(
    renderer: Any,
    background_bgr: np.ndarray,
    verts_t_v_3: np.ndarray,
    origcam_t_4: np.ndarray,
    gt_emg_norm_8_t: np.ndarray,
    pred_emg_norm_8_t: np.ndarray,
    out_mp4: Path,
    fps: int,
    plot_width: int,
    plot_height: int,
) -> None:
    cell_frames = _render_sequence_cells(
        renderer=renderer,
        background_bgr=background_bgr,
        verts_t_v_3=verts_t_v_3,
        origcam_t_4=origcam_t_4,
        gt_emg_norm_8_t=gt_emg_norm_8_t,
        pred_emg_norm_8_t=pred_emg_norm_8_t,
        fps=fps,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    _write_video_mp4(cell_frames, out_mp4, fps=fps)


def _render_sequence_cells(
    renderer: Any,
    background_bgr: np.ndarray,
    verts_t_v_3: np.ndarray,
    origcam_t_4: np.ndarray,
    gt_emg_norm_8_t: np.ndarray,
    pred_emg_norm_8_t: np.ndarray,
    fps: int,
    plot_width: int,
    plot_height: int,
) -> list[np.ndarray]:
    t = gt_emg_norm_8_t.shape[1]
    panel_gt = _render_emg_panel(gt_emg_norm_8_t, plot_width, plot_height, "GT EMG", vmax=1.0)
    panel_pred = _render_emg_panel(pred_emg_norm_8_t, plot_width, plot_height, "Pred EMG", vmax=1.0)
    panel_gt = cv2.cvtColor(panel_gt, cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(panel_pred, cv2.COLOR_RGB2BGR)

    frames: list[np.ndarray] = []
    for i in range(t):
        verts = verts_t_v_3[i]
        cam = origcam_t_4[i]
        gt_img, _ = renderer.render(
            flag="False",
            current_path="/tmp/mia_vis_gt",
            img=background_bgr,
            verts=verts,
            emg_values=gt_emg_norm_8_t[:, i],
            cam=cam,
            front=True,
            pred=False,
        )
        pred_img, _ = renderer.render(
            flag="False",
            current_path="/tmp/mia_vis_pred",
            img=background_bgr,
            verts=verts,
            emg_values=pred_emg_norm_8_t[:, i],
            cam=cam,
            front=True,
            pred=True,
        )

        gt_row = np.concatenate([gt_img, panel_gt], axis=1)
        pred_row = np.concatenate([pred_img, panel_pred], axis=1)
        frame = np.concatenate([gt_row, pred_row], axis=0)
        frames.append(frame)
    return frames


def _pad_time_first(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    pad_len = target_len - arr.shape[0]
    last = arr[-1:]
    pad = np.repeat(last, pad_len, axis=0)
    return np.concatenate([arr, pad], axis=0)


def _pad_emg_8_t(emg_8_t: np.ndarray, target_len: int) -> np.ndarray:
    if emg_8_t.shape[1] == target_len:
        return emg_8_t
    if emg_8_t.shape[1] > target_len:
        return emg_8_t[:, :target_len]
    pad_len = target_len - emg_8_t.shape[1]
    last = emg_8_t[:, -1:]
    pad = np.repeat(last, pad_len, axis=1)
    return np.concatenate([emg_8_t, pad], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/MIADatasetOfficial",
    )
    parser.add_argument("--phase", type=str, default="val")
    parser.add_argument("--n_per_exercise", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_exercises", type=int, default=-1)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth",
    )
    parser.add_argument(
        "--smpl_src",
        type=str,
        default="/data/litengmo/HSMR/SMPL_models/models/smpl",
    )
    parser.add_argument(
        "--vibe_dst",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/musclesinaction/vibe_data",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/output/vis_infer_final",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--render_width", type=int, default=640)
    parser.add_argument("--render_height", type=int, default=360)
    parser.add_argument("--plot_width", type=int, default=420)
    parser.add_argument("--plot_height", type=int, default=360)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    smpl_src_dir = Path(args.smpl_src)
    vibe_dst_dir = Path(args.vibe_dst)

    samples = _scan_samples(dataset_root, args.phase)
    samples_by_exercise: dict[str, list[SampleRef]] = {}
    for s in samples:
        samples_by_exercise.setdefault(s.exercise, []).append(s)

    exercise_names = sorted(samples_by_exercise.keys())
    if args.max_exercises is not None and args.max_exercises > 0:
        exercise_names = exercise_names[: args.max_exercises]

    selected: list[SampleRef] = []
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
        random.shuffle(cand)
        selected.extend(cand[: args.n_per_exercise])

    selected_by_exercise: dict[str, list[SampleRef]] = {}
    for s in selected:
        selected_by_exercise.setdefault(s.exercise, []).append(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "phase": args.phase,
                "seed": args.seed,
                "n_per_exercise": args.n_per_exercise,
                "checkpoint": str(checkpoint_path),
                "selected": [s.key for s in selected],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"Found exercises: {len(samples_by_exercise)}")
        print(f"Selected samples: {len(selected)}")
        return

    try:
        import torch
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyTorch is required for inference. Activate your environment and ensure `import torch` works."
        ) from e

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    _ensure_vibe_data(smpl_src_dir, vibe_dst_dir)

    from musclesinaction.vis.renderer import Renderer

    renderer = Renderer(
        resolution=(int(args.render_width), int(args.render_height)),
        orig_img=True,
        wireframe=False,
    )

    background = cv2.imread("backplain.png")
    if background is None:
        raise FileNotFoundError("backplain.png not found in current working directory")
    background = cv2.resize(background, (int(args.render_width), int(args.render_height)))
    background = background.astype(np.uint8)

    model = _load_model(checkpoint_path, device=device)

    for exercise, samples_in_ex in sorted(selected_by_exercise.items(), key=lambda x: x[0]):
        prepared: list[dict[str, Any]] = []
        max_t = 0
        for sample in samples_in_ex:
            arrays = _load_sample_arrays(sample)
            condval = _subject_to_condval(sample.subject)
            pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval=condval, device=device)
            gt_emg_8_t = arrays["emg_8_t"]
            gt_emg_norm = _normalize_emg(gt_emg_8_t, sample.subject)
            pred_emg_norm = _normalize_emg(pred_emg_8_t, sample.subject)
            t = gt_emg_norm.shape[1]
            max_t = max(max_t, t)
            prepared.append(
                {
                    "sample": sample,
                    "verts": arrays["verts_t_v_3"],
                    "cam": arrays["origcam_t_4"],
                    "gt_emg": gt_emg_norm,
                    "pred_emg": pred_emg_norm,
                }
            )

        if not prepared:
            continue

        for item in prepared:
            item["verts"] = _pad_time_first(item["verts"], max_t)
            item["cam"] = _pad_time_first(item["cam"], max_t)
            item["gt_emg"] = _pad_emg_8_t(item["gt_emg"], max_t)
            item["pred_emg"] = _pad_emg_8_t(item["pred_emg"], max_t)

        cell_streams: list[list[np.ndarray]] = []
        for item in prepared:
            cell_frames = _render_sequence_cells(
                renderer=renderer,
                background_bgr=background,
                verts_t_v_3=item["verts"],
                origcam_t_4=item["cam"],
                gt_emg_norm_8_t=item["gt_emg"],
                pred_emg_norm_8_t=item["pred_emg"],
                fps=int(args.fps),
                plot_width=int(args.plot_width),
                plot_height=int(args.plot_height),
            )
            cell_streams.append(cell_frames)

        cell_h, cell_w = cell_streams[0][0].shape[:2]
        out_mp4 = out_dir / exercise / f"grid_n{len(prepared)}.mp4"
        writer = _open_video_writer(out_mp4, fps=int(args.fps), frame_size_wh=(cell_w * len(cell_streams), cell_h))
        try:
            for i in range(max_t):
                cols = [cell_streams[c][i] for c in range(len(cell_streams))]
                frame = np.concatenate(cols, axis=1)
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    main()
