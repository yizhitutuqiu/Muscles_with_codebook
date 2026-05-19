from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _iter_loader(loader: DataLoader, *, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(loader, desc=desc, unit="batch")
    except Exception:
        return loader


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


def build_retrieval_db(
    *,
    loader: DataLoader,
    task: str,
    device: torch.device,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    max_samples: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    seen = 0

    for batch in _iter_loader(loader, desc="retrieval_db"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)

        b = int(joints3d.shape[0])

        if str(task).strip().lower() == "pose2emg":
            gt_bt8 = emg.permute(0, 2, 1).contiguous()
            x = joints3d.reshape(b, -1).contiguous()
            y = gt_bt8.reshape(b, -1).contiguous()
        else:
            emg_bt8 = emg.permute(0, 2, 1).contiguous()
            x = emg_bt8.reshape(b, -1).contiguous()
            y = joints3d.reshape(b, -1).contiguous()

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


def eval_retrieval_on_filelist(
    *,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    loader: DataLoader,
    task: str,
    device: torch.device,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    train_chunk_size: int,
    muscle_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    pred_all: List[np.ndarray] = []
    gt_all: List[np.ndarray] = []
    per_sequence_rows: List[Dict[str, Any]] = []
    sample_counter = 0
    task = str(task).strip().lower()

    for batch in _iter_loader(loader, desc="retrieval"):
        joints3d = torch.as_tensor(batch["3dskeleton"], device=device, dtype=torch.float32)
        emg = torch.as_tensor(batch["emg_values"], device=device, dtype=torch.float32)
        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)

        if task == "pose2emg":
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

            names = list(muscle_names) if muscle_names is not None else []
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
                for ch in range(min(8, len(names))):
                    row[f"rmse_{names[ch]}"] = float(math.sqrt(np.mean(seq_sq_err[:, ch])))
                per_sequence_rows.append(row)
                sample_counter += 1
        else:
            emg_bt8 = emg.permute(0, 2, 1).contiguous()
            b, t, _ = emg_bt8.shape
            xq = emg_bt8.reshape(b, -1).contiguous()
            with torch.no_grad():
                ypred_flat = _predict_retrieval(xtr=xtr, ytr=ytr, xq=xq, train_chunk_size=train_chunk_size)
            pred_joints = ypred_flat.view(b, t, 25, 3)

            pred_np = pred_joints.detach().cpu().numpy()
            gt_np = joints3d.detach().cpu().numpy()
            pred_all.append(pred_np.reshape(b * t, 25, 3))
            gt_all.append(gt_np.reshape(b * t, 25, 3))

            filepaths = _get_filepaths_from_batch(batch, int(gt_np.shape[0]))
            for i in range(int(gt_np.shape[0])):
                diff = pred_np[i] - gt_np[i]
                seq_sq_err = np.square(diff)
                seq_abs_err = np.abs(diff)
                row = {
                    "eval_index": sample_counter,
                    "filepath": filepaths[i],
                    "frames": int(gt_np.shape[1]),
                    "mse": float(np.mean(seq_sq_err)),
                    "rmse": float(math.sqrt(np.mean(seq_sq_err))),
                    "mae": float(np.mean(seq_abs_err)),
                }
                per_sequence_rows.append(row)
                sample_counter += 1

    pred_cat = np.concatenate(pred_all, axis=0)
    gt_cat = np.concatenate(gt_all, axis=0)
    return pred_cat, gt_cat, per_sequence_rows

