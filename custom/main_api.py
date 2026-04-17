"""
Stage2 总模型推理 API：任意长度 3D 关节序列 -> EMG 序列。

内部按 clip_len（默认 30）滑窗，步长为 inference.step；重叠区域对预测取平均。
配置见 custom/configs/infer.yaml。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml

_MIA_ROOT = Path(__file__).resolve().parents[2]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.tools.eval_stage2_pose2emg_official_metrics import (  # noqa: E402
    _build_stage2_from_ckpt,
    _emg_standardizer_std,
    _root_center_joints3d,
)
from custom.utils.path_utils import get_musclesinaction_repo_root  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _window_starts(total_frames: int, clip_len: int, step: int) -> List[int]:
    """生成滑窗起点，保证首尾覆盖；T < clip_len 时仅 [0]（由调用方 padding）。"""
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if clip_len <= 0 or step <= 0:
        raise ValueError(f"clip_len and step must be positive, got clip_len={clip_len}, step={step}")
    if total_frames <= clip_len:
        return [0]
    last_start = total_frames - clip_len
    starts = list(range(0, last_start + 1, step))
    if starts[-1] < last_start:
        starts.append(last_start)
    return starts


def _to_tensor_joints3d(
    joints3d: Union[np.ndarray, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(T,25,3) 或 (T,75) -> (T,25,3) float tensor on device."""
    if torch.is_tensor(joints3d):
        x = joints3d.to(device=device, dtype=dtype)
    else:
        x = torch.as_tensor(np.asarray(joints3d), device=device, dtype=dtype)
    if x.ndim == 2 and x.shape[-1] == 75:
        x = x.reshape(-1, 25, 3)
    if x.ndim != 3 or x.shape[-2:] != (25, 3):
        raise ValueError(f"Expected joints3d (T,25,3) or (T,75), got {tuple(x.shape)}")
    return x


def _pad_clip_right(x: torch.Tensor, clip_len: int) -> torch.Tensor:
    """x: (T,25,3), T < clip_len 时在时间维右侧重复最后一帧 pad 到 clip_len。"""
    t = int(x.shape[0])
    if t >= clip_len:
        return x
    pad_n = clip_len - t
    last = x[-1:].expand(pad_n, -1, -1).clone()
    return torch.cat([x, last], dim=0)


class Stage2EMGPredictor:
    """
    加载 Stage2 checkpoint，对任意长度关节序列做滑窗推理并融合重叠区域。

    用法:
        pred = Stage2EMGPredictor.from_yaml("custom/configs/infer.yaml")
        emg = pred.predict(joints3d)  # (T,25,3) -> (T,8) numpy
    """

    def __init__(
        self,
        *,
        infer_cfg: Dict[str, Any],
        config_path: Optional[Path] = None,
    ) -> None:
        self.mia_root = get_musclesinaction_repo_root()
        self.infer_cfg = infer_cfg
        self.config_path = config_path

        runtime = infer_cfg.get("runtime", {}) or {}
        device_str = str(runtime.get("device", "cuda"))
        if device_str == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)

        stage2_cfg = infer_cfg.get("stage2", {}) or {}
        ckpt = Path(str(stage2_cfg.get("checkpoint", ""))).expanduser()
        if not ckpt.is_absolute():
            ckpt = (self.mia_root / ckpt).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Stage2 checkpoint not found: {ckpt}")

        s1_override = infer_cfg.get("stage1", {}) or {}
        s1_path = s1_override.get("checkpoint")
        stage1_override = None
        if s1_path:
            p = Path(str(s1_path)).expanduser()
            stage1_override = p if p.is_absolute() else (self.mia_root / p).resolve()

        self.model, self.train_cfg, self.stage1_path, self.stage1 = _build_stage2_from_ckpt(
            ckpt, self.device, stage1_override_ckpt=stage1_override
        )
        self.model.eval()

        inf = infer_cfg.get("inference", {}) or {}
        self.clip_len = int(inf.get("clip_len", 30))
        self.step = int(inf.get("step", 15))
        self.inference_batch_size = max(1, int(inf.get("inference_batch_size", 8)))
        self.joints3d_root_center = bool(inf.get("joints3d_root_center", True))
        self.joints3d_root_index = int(inf.get("joints3d_root_index", 8))

        pred_mode = str(self.train_cfg.get("model", {}).get("emg_pred_mode", "full")).strip().lower()
        emg_normalize_target = bool(self.train_cfg.get("data", {}).get("emg_normalize_target", False))
        stage1_emg = getattr(self.stage1, "emg", None)
        emg_std = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
        self._use_emg_denorm = pred_mode == "full" and emg_normalize_target and emg_std is not None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Stage2EMGPredictor":
        p = Path(yaml_path).expanduser()
        if not p.is_absolute():
            p = (get_musclesinaction_repo_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"infer config not found: {p}")
        return cls(infer_cfg=_load_yaml(p), config_path=p)

    def _denorm_emg(self, emg_pred: torch.Tensor) -> torch.Tensor:
        if not self._use_emg_denorm:
            return emg_pred
        stdzr = self.stage1.emg.standardizer
        std = _emg_standardizer_std(stdzr).to(emg_pred.device)
        return emg_pred * std + stdzr.mean.to(emg_pred.device)

    @torch.no_grad()
    def predict(
        self,
        joints3d: Union[np.ndarray, torch.Tensor],
        *,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        输入任意长度 3D 关节序列，输出同长度 EMG (8 通道)。

        :param joints3d: (T, 25, 3) 或 (T, 75)，世界坐标或已 root-center 均可（由配置 joints3d_root_center 决定）
        :param return_torch: True 则返回 torch.Tensor (T,8) 在 self.device
        """
        x = _to_tensor_joints3d(joints3d, device=self.device, dtype=torch.float32)
        t_total = int(x.shape[0])
        if t_total == 0:
            raise ValueError("joints3d has zero frames")

        L = self.clip_len
        starts = _window_starts(t_total, L, self.step)

        clips: List[torch.Tensor] = []
        for s in starts:
            seg = x[s : s + L]
            if seg.shape[0] < L:
                seg = _pad_clip_right(seg, L)
            clips.append(seg)

        emg_sum = torch.zeros(t_total, 8, device=self.device, dtype=torch.float32)
        emg_cnt = torch.zeros(t_total, device=self.device, dtype=torch.float32)

        bs = self.inference_batch_size
        for i in range(0, len(clips), bs):
            batch_clips = clips[i : i + bs]
            b = len(batch_clips)
            batch = torch.stack(batch_clips, dim=0)  # (b, L, 25, 3)
            if self.joints3d_root_center:
                batch = _root_center_joints3d(batch, self.joints3d_root_index)
            out = self.model(batch)
            pred = self._denorm_emg(out["emg_pred"])  # (b, L, 8)

            for j in range(b):
                s = starts[i + j]
                length = min(L, t_total - s)
                if length <= 0:
                    continue
                emg_sum[s : s + length] += pred[j, :length]
                emg_cnt[s : s + length] += 1.0

        emg_cnt = torch.clamp(emg_cnt, min=1e-6)
        emg_out = emg_sum / emg_cnt.unsqueeze(-1)

        if return_torch:
            return emg_out
        return emg_out.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_gvhmr_stride(
        self,
        joints3d: Union[np.ndarray, torch.Tensor],
        *,
        stride: int,
    ) -> np.ndarray:
        """
        与 GVHMR `predict_emg_from_hmr4d._predict_emg_from_joints25` 相同的滑窗策略：
        ``for start in range(0, T, stride)``，每窗 ``clip_len`` 帧，不足则右侧重复末帧 pad，
        重叠区域累加后除以次数；最后一窗后若 ``end >= T`` 则结束循环。

        注意：调用方应先对全序列做与 GVHMR 一致的 root-center（MidHip=8），再传入本函数；
        本函数内不再做全序列 root-center，仅在每窗上按 Stage2 训练口径调用 ``model``（内部可对每窗再做 root-center，对已 centered 的帧为恒等）。
        """
        joints25 = _to_tensor_joints3d(joints3d, device=self.device, dtype=torch.float32)
        t = int(joints25.shape[0])
        if t == 0:
            return np.zeros((0, 8), dtype=np.float32)
        if stride <= 0:
            raise ValueError("stride must be > 0")
        step = int(self.clip_len)
        acc = torch.zeros((t, 8), device=self.device, dtype=torch.float32)
        cnt = torch.zeros((t, 1), device=self.device, dtype=torch.float32)

        for start in range(0, t, stride):
            end = start + step
            win = joints25[start:end]
            valid = int(win.shape[0])
            if valid < step:
                pad = win[-1:].expand(step - valid, 25, 3)
                win = torch.cat([win, pad], dim=0)
            batch = win.unsqueeze(0)
            if self.joints3d_root_center:
                batch = _root_center_joints3d(batch, self.joints3d_root_index)
            out = self.model(batch)
            pred = self._denorm_emg(out["emg_pred"])
            y = pred[0][:valid]
            acc[start : start + valid] += y
            cnt[start : start + valid] += 1.0
            if end >= t:
                break

        emg = (acc / torch.clamp_min(cnt, 1.0)).detach().cpu().numpy().astype(np.float32)
        return emg


def predict_emg_from_joints3d(
    joints3d: Union[np.ndarray, torch.Tensor],
    *,
    infer_yaml: Union[str, Path, None] = None,
    return_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    便捷函数：默认读取 custom/configs/infer.yaml，一次推理。
    若需多次调用请使用 Stage2EMGPredictor.from_yaml 并复用实例。
    """
    if infer_yaml is None:
        infer_yaml = get_musclesinaction_repo_root() / "custom" / "configs" / "infer.yaml"
    pred = Stage2EMGPredictor.from_yaml(infer_yaml)
    return pred.predict(joints3d, return_torch=return_torch)


__all__ = [
    "Stage2EMGPredictor",
    "predict_emg_from_joints3d",
    "_window_starts",
]
