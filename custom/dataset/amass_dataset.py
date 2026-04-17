"""
AMASS dataset for Stage1 frame codebook pre-training (j3d only).

Loads AMASS npz (trans, root_orient, pose_body, betas), runs SMPL to get 24 joints,
converts to 25 joints with MidHip at index 8 and root-centering at 8,
then reorders to MIA (OpenPose Body25) joint order so pretrained model matches
MIA/finetune/evals. Returns batches in the same format as MyMuscleDataset
(3dskeleton, emg_values) so the same train loop can be used.

Important: MIA uses OpenPose Body25 order (0=Nose, 1=Neck, ..., 8=MidHip, 9=RHip, ...).
SMPL 24→25 gives a different order (0=Pelvis, 1=L_Hip, 2=R_Hip, ..., 8=MidHip, ...).
Without reordering, AMASS-pretrained model outputs would be wrong on MIA (e.g. "一团乱麻").
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# SMPL 24 joint order (smplx): 0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee,
# 6 Spine2, 7 L_Ankle, 8 R_Ankle, 9 Spine3, 10 L_Foot, 11 R_Foot, 12 Neck, 13 L_Collar,
# 14 R_Collar, 15 Head, 16 L_Shoulder, 17 R_Shoulder, 18 L_Elbow, 19 R_Elbow,
# 20 L_Wrist, 21 R_Wrist, 22 L_Hand, 23 R_Hand.
# We build 25: J25[0:8]=J24[0:8], J25[8]=mid_hip=(J24[1]+J24[2])/2, J25[9:25]=J24[8:24].
# Root for MIA is index 8 (MidHip).
MID_HIP_SMPL_LEFT = 1
MID_HIP_SMPL_RIGHT = 2
MIA_ROOT_INDEX = 8

# Map from MIA (OpenPose Body25) index to our SMPL-derived 25 index.
# MIA order: 0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist, 5 LShoulder, 6 LElbow, 7 LWrist,
# 8 MidHip, 9 RHip, 10 RKnee, 11 RAnkle, 12 LHip, 13 LKnee, 14 LAnkle, 15-18 REye/LEye/REar/LEar,
# 19-21 LBigToe/LSmallToe/LHeel, 22-24 RBigToe/RSmallToe/RHeel.
# SMPL has no Nose/Eye/Ear; we use Head(15). Feet: use L_Foot(10)/R_Foot(11).
SMPL25_TO_MIA_OP25: tuple[int, ...] = (
    15, 12, 17, 19, 21, 16, 18, 20, 8, 2, 5, 9, 1, 4, 7,
    15, 15, 15, 15, 10, 10, 10, 11, 11, 11,
)


def _smpl_forward_np(
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    betas: np.ndarray,
    smpl_model_path: str,
    gender: str = "neutral",
) -> np.ndarray:
    """Run SMPL forward; returns joints (T, 24, 3) in meters."""
    try:
        import smplx
    except ImportError:
        raise ImportError("Please install smplx: pip install smplx")

    device = torch.device("cpu")
    model_path = Path(smpl_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SMPL model path not found: {model_path}")
    model_dir = model_path.parent if model_path.suffix == ".pkl" else model_path
    # smplx expects model_dir/smpl/SMPL_*.pkl; if pkl is directly in model_dir, use parent + symlink
    smpl_sub = model_dir / "smpl"
    if (model_dir / "SMPL_NEUTRAL.pkl").exists() and not smpl_sub.exists():
        parent = model_dir.parent
        link = parent / "smpl"
        if not link.exists():
            try:
                link.symlink_to(model_dir.resolve(), target_is_directory=True)
            except OSError:
                raise FileNotFoundError(
                    f"smplx expects model_dir/smpl/ with SMPL_*.pkl. Your pkl is in {model_dir}. "
                    f"Create: mkdir -p {smpl_sub} then copy or symlink SMPL_*.pkl into it, "
                    f"or run: ln -s {model_dir} {link}"
                ) from None
        model_dir = parent

    # AMASS uses body pose 63; smplx expects body_pose (21*3)=63
    num_frames = trans.shape[0]
    if root_orient.ndim == 1:
        root_orient = root_orient.reshape(1, 3)
    if pose_body.ndim == 1:
        pose_body = pose_body.reshape(1, 63)
    if trans.ndim == 1:
        trans = trans.reshape(1, 3)
    if betas.ndim == 0:
        betas = betas.reshape(1, -1)
    if betas.ndim == 1:
        betas = np.tile(betas.reshape(1, -1), (num_frames, 1))
    # SMPL typically uses 10 betas; AMASS may have 16
    if betas.shape[-1] > 10:
        betas = betas[..., :10]

    # smplx SMPL uses 23 body joints (69 dims); AMASS pose_body is 21 joints (63 dims). Pad to 69.
    if pose_body.shape[-1] == 63:
        pose_body = np.concatenate([pose_body, np.zeros((*pose_body.shape[:-1], 6), dtype=pose_body.dtype)], axis=-1)

    body_model = smplx.create(
        str(model_dir),
        model_type="smpl",
        gender=gender,
        batch_size=num_frames,
    )
    body_pose = torch.from_numpy(pose_body).float().to(device)
    global_orient = torch.from_numpy(root_orient).float().to(device)
    transl = torch.from_numpy(trans).float().to(device)
    betas_t = torch.from_numpy(betas[:num_frames]).float().to(device)

    out = body_model(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        betas=betas_t,
    )
    # smplx may return 24 or more joints (e.g. 46 with J_regressor_extra); use first 24 body joints only
    joints_all = out.joints.detach().cpu().numpy()
    joints_24 = joints_all[..., :24, :]
    return joints_24


def joints24_to_25_root_centered(joints_24: np.ndarray, root_index: int = MIA_ROOT_INDEX) -> np.ndarray:
    """
    Convert SMPL 24 joints to 25 with MidHip at index 8, then root-center at root_index.
    joints_24: (T, 24, 3) or (24, 3).
    """
    if joints_24.ndim == 2:
        joints_24 = joints_24[np.newaxis, ...]
    T = joints_24.shape[0]
    mid_hip = (joints_24[:, MID_HIP_SMPL_LEFT] + joints_24[:, MID_HIP_SMPL_RIGHT]) / 2.0
    j25 = np.concatenate(
        [
            joints_24[:, :8],
            mid_hip[:, np.newaxis],
            joints_24[:, 8:],
        ],
        axis=1,
    )
    root = j25[:, root_index : root_index + 1]
    j25 = j25 - root
    if j25.shape[0] == 1:
        j25 = j25[0]
    return j25.astype(np.float32)


def _smpl25_to_mia_order(j25: np.ndarray) -> np.ndarray:
    """
    Reorder SMPL-derived 25 joints to MIA (OpenPose Body25) order.
    j25: (T, 25, 3) or (25, 3). Returns same shape.
    """
    idx = np.asarray(SMPL25_TO_MIA_OP25, dtype=np.intp)
    if j25.ndim == 2:
        return j25[idx].astype(np.float32)
    return j25[:, idx, :].astype(np.float32)


class AmassCodebookDataset(Dataset):
    """
    Dataset over AMASS npz files for frame codebook pre-training.
    Each sample is a clip of `step` consecutive frames; returns the same keys as
    MyMuscleDataset (3dskeleton, emg_values with zeros) so the same collate and
    _prepare_batch can be used.
    """

    def __init__(
        self,
        filelist_path: str,
        step: int = 30,
        smpl_model_path: Optional[str] = None,
        joints3d_root_center: bool = True,
        joints3d_root_index: int = 8,
        percent: float = 1.0,
        max_samples: Optional[int] = None,
    ):
        """
        :param filelist_path: Path to a .txt with one npz path per line (absolute or relative to cwd).
        :param step: Number of frames per clip (same as MIA step).
        :param smpl_model_path: Directory containing SMPL (e.g. golf_third_party/base_data with SMPL_NEUTRAL.pkl).
        :param joints3d_root_center: Whether to root-center joints (same as MIA).
        :param joints3d_root_index: Root joint index (8 = MidHip, same as MIA).
        :param percent: Fraction of filelist to use (1.0 = all).
        :param max_samples: Cap number of samples (None = no cap).
        """
        self.step = int(step)
        self.joints3d_root_center = joints3d_root_center
        self.joints3d_root_index = int(joints3d_root_index)
        self.smpl_model_path = smpl_model_path
        self._smpl_cache = {}

        with open(filelist_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if max_samples is not None:
            lines = lines[: int(max_samples)]
        self.file_paths = lines
        n = int(len(self.file_paths) * float(percent))
        self.file_paths = self.file_paths[:n]
        self._clip_starts: list[tuple[str, int]] = []
        for fp in self.file_paths:
            path = Path(fp)
            if not path.is_absolute():
                path = Path(os.getcwd()) / path
            if not path.exists():
                continue
            with np.load(path, allow_pickle=True) as data:
                num_frames = data["trans"].shape[0]
            for start in range(0, max(1, num_frames - self.step + 1), self.step):
                self._clip_starts.append((str(path), start))
        self._clip_starts = self._clip_starts

    def __len__(self) -> int:
        return len(self._clip_starts)

    def _get_joints25_for_clip(self, npz_path: str, start: int) -> np.ndarray:
        """Load one npz, run SMPL for frames [start : start+step], return (step, 25, 3) root-centered."""
        data = np.load(npz_path, allow_pickle=True)
        end = start + self.step
        trans = data["trans"][start:end]
        root_orient = data["root_orient"][start:end]
        pose_body = data["pose_body"][start:end]
        betas = data["betas"]
        g = data.get("gender", np.array("neutral"))
        gender = str(g.item() if hasattr(g, "item") else g).strip().lower()
        if gender not in ("male", "female", "neutral"):
            gender = "neutral"

        if self.smpl_model_path is None:
            raise RuntimeError("AmassCodebookDataset requires smpl_model_path")

        joints_24 = _smpl_forward_np(
            trans, root_orient, pose_body, betas, self.smpl_model_path, gender=gender
        )
        if self.joints3d_root_center:
            j25 = joints24_to_25_root_centered(
                joints_24, root_index=self.joints3d_root_index
            )
        else:
            mid_hip = (joints_24[:, MID_HIP_SMPL_LEFT] + joints_24[:, MID_HIP_SMPL_RIGHT]) / 2.0
            j25 = np.concatenate(
                [joints_24[:, :8], mid_hip[:, np.newaxis], joints_24[:, 8:]], axis=1
            ).astype(np.float32)
        if j25.shape[0] < self.step:
            pad = np.zeros((self.step - j25.shape[0], 25, 3), dtype=np.float32)
            j25 = np.concatenate([j25, pad], axis=0)
        j25 = j25[: self.step]
        # Reorder to MIA (OpenPose Body25) so pretrain matches MIA finetune/evals.
        j25 = _smpl25_to_mia_order(j25)
        return j25

    def __getitem__(self, index: int) -> dict:
        npz_path, start = self._clip_starts[index]
        joints25 = self._get_joints25_for_clip(npz_path, start)
        T, J, C = joints25.shape
        emg_zeros = np.zeros((8, T), dtype=np.float32)
        return {
            "3dskeleton": joints25,
            "emg_values": emg_zeros,
            "pose": np.zeros((T, 72), dtype=np.float32),
            "filepath": npz_path,
        }
