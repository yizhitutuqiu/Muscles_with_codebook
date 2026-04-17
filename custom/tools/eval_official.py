from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# File path:
#   .../golf_third_party/musclesinaction/custom/tools/eval_official.py
_MIA_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.utils.mia_filelist import build_mia_train_filelist  # noqa: E402
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


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _default_checkpoint(mia_root: Path) -> Path:
    return (
        mia_root
        / "pretrained-checkpoints"
        / "generalization_new_cond_clean_posetoemg"
        / "model_100.pth"
    )


def _maybe_build_filelist(
    *,
    mia_root: Path,
    split: str,
    out_txt: Path,
    max_samples: int | None,
    require_files: Sequence[str],
) -> Path:
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


def _prepare_eval_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
    emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
    cond = torch.as_tensor(batch["condval"], device=device, dtype=torch.float32)

    if joints3d.ndim != 4:
        raise ValueError(f"Expected 3dskeleton (B,T,25,3), got {tuple(joints3d.shape)}")
    if emg.ndim != 3:
        raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")
    if cond.ndim == 1:
        cond = cond.unsqueeze(1)

    b, t, j, c = joints3d.shape
    if (j, c) != (25, 3):
        raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
    if emg.shape != (b, 8, t):
        raise ValueError(f"Expected emg_values (B,8,T), got {tuple(emg.shape)}")

    skeleton = joints3d.reshape(b, t, 75)
    gt_bt8 = emg.permute(0, 2, 1).contiguous()
    return skeleton, cond, gt_bt8, b, t


@torch.no_grad()
def _run_eval(
    *,
    ckpt_path: Path,
    filelist_path: Path,
    split: str,
    std: bool,
    cond: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    out_root: Path,
) -> Dict[str, Any]:
    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    from musclesinaction.dataloader.data import MyMuscleDataset, _seed_worker  # type: ignore

    model, payload, model_args = _load_official_model(ckpt_path, device)
    cond_enabled = bool(cond)

    dset = MyMuscleDataset(
        str(filelist_path),
        _NullLogger(),
        split,
        percent=1.0,
        step=int(model_args.get("step", 30)),
        std="True" if std else "False",
        cond="True" if cond_enabled else "False",
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

    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    batch_counter = 0
    sample_counter = 0

    for batch in tqdm(loader, desc="eval", unit="batch"):
        skeleton, cond_bt, gt_bt8, b, t = _prepare_eval_batch(batch, device)
        pred_b8t = model(skeleton, cond_bt)
        if pred_b8t.ndim != 3 or pred_b8t.shape != (b, 8, t):
            raise RuntimeError(f"Unexpected official model output shape: {tuple(pred_b8t.shape)}")

        pred_bt8 = pred_b8t.permute(0, 2, 1).contiguous().detach().cpu().numpy()
        gt_bt8_np = gt_bt8.detach().cpu().numpy()
        pred_all.append(pred_bt8.reshape(b * t, 8))
        gt_all.append(gt_bt8_np.reshape(b * t, 8))

        filepaths = _get_filepaths_from_batch(batch, b)
        for i in range(b):
            seq_sq_err = np.square(pred_bt8[i] - gt_bt8_np[i])
            seq_abs_err = np.abs(pred_bt8[i] - gt_bt8_np[i])
            row = {
                "eval_index": sample_counter,
                "filepath": filepaths[i],
                "frames": t,
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

    run_dir = out_root / ckpt_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    per_sequence_csv = run_dir / "per_sequence.csv"
    summary_json = run_dir / "summary.json"
    pred_npz = run_dir / "pred_vs_gt.npz"

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
        "split": split,
        "filelist": str(filelist_path),
        "std": bool(std),
        "cond": bool(cond_enabled),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "num_batches": int(batch_counter),
        "num_sequences": int(len(per_sequence_rows)),
        "model_args": model_args,
        **metrics,
        "outputs": {
            "per_sequence_csv": str(per_sequence_csv),
            "pred_vs_gt_npz": str(pred_npz),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez_compressed(pred_npz, pred_emg=pred_cat.astype(np.float32), gt_emg=gt_cat.astype(np.float32))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the official musclesinaction pose->EMG model with the same summary format as custom evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Official checkpoint path. Default: pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth",
    )
    parser.add_argument("--split", type=str, default="val", help="train / val / test. If test missing, fallback to val.")
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help="Optional txt filelist to evaluate. Useful for reproducing official valRunning/valSubjectX protocols.",
    )
    parser.add_argument(
        "--std",
        type=_str2bool,
        default=False,
        help="Whether to load emgvaluesstd.npy like official scripts. Default: false.",
    )
    parser.add_argument(
        "--cond",
        type=_str2bool,
        default=True,
        help="Whether to enable conditioning (condval) during evaluation. Default: true. For nocond checkpoints, set --cond false.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Only used when auto-building filelist.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: custom/output/eval_official/<timestamp>/",
    )
    args = parser.parse_args()

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    split = _resolve_split(args.split, mia_root)
    ckpt_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _default_checkpoint(mia_root).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Official checkpoint not found: {ckpt_path}")

    if args.filelist:
        filelist_path = Path(args.filelist).expanduser().resolve()
        if not filelist_path.exists():
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")
    else:
        # 与 eval_stage2_pose2emg_official_metrics.py 默认一致，便于对比：miaofficial_{split}_eval.txt
        filelist_name = f"miaofficial_{split}_eval.txt"
        filelist_path = mia_root / "custom" / "tools" / "datasetsplits" / filelist_name
        _maybe_build_filelist(
            mia_root=mia_root,
            split=split,
            out_txt=filelist_path,
            max_samples=args.max_samples,
            require_files=["emgvalues.npy" if not args.std else "emgvaluesstd.npy", "joints3d.npy"],
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_root = Path(args.out_dir).expanduser().resolve()
    else:
        out_root = (mia_root / "custom" / "output" / "eval_official" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        raise RuntimeError(
            "官方 modelposetoemg.py 内部硬编码了 torch.cuda.FloatTensor，当前评测脚本只支持 --device cuda。"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is not available, but official evaluator currently requires cuda.")

    print("[EvalOfficial] split =", split)
    print("[EvalOfficial] checkpoint =", ckpt_path)
    print("[EvalOfficial] filelist =", filelist_path)
    print("[EvalOfficial] std =", bool(args.std))
    print("[EvalOfficial] cond =", bool(args.cond))
    print("[EvalOfficial] device =", device)
    print("[EvalOfficial] out_root =", out_root)

    summary = _run_eval(
        ckpt_path=ckpt_path,
        filelist_path=filelist_path,
        split=split,
        std=bool(args.std),
        cond=bool(args.cond),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        out_root=out_root,
    )

    summary_csv = out_root / "summary.csv"
    _write_csv_rows(
        summary_csv,
        [
            {
                "checkpoint": summary["checkpoint"],
                "split": summary["split"],
                "filelist": summary["filelist"],
                "std": summary["std"],
                "cond": summary["cond"],
                "num_sequences": summary["num_sequences"],
                "num_frames": summary["num_frames"],
                "official_global_mse": summary["official_global_mse"],
                "official_global_rmse": summary["official_global_rmse"],
                "official_global_mae": summary["official_global_mae"],
                "summary_json": str((out_root / ckpt_path.stem / "summary.json").resolve()),
            }
        ],
        [
            "checkpoint",
            "split",
            "filelist",
            "std",
            "cond",
            "num_sequences",
            "num_frames",
            "official_global_mse",
            "official_global_rmse",
            "official_global_mae",
            "summary_json",
        ],
    )

    print(
        "[EvalOfficialResult]",
        {
            "checkpoint": ckpt_path.name,
            "official_global_rmse": summary["official_global_rmse"],
            "official_global_mse": summary["official_global_mse"],
            "num_sequences": summary["num_sequences"],
        },
    )
    print("[EvalOfficial] saved_summary_csv =", summary_csv)


if __name__ == "__main__":
    main()
