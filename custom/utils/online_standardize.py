from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class OnlineStandardizeConfig:
    eps: float = 1e-6
    momentum: float = 0.01  # EMA update; smaller = slower changes
    clip_std_min: float = 1e-3


class OnlineStandardizer(nn.Module):
    """
    Online per-feature standardization using EMA mean/var.

    This is intentionally simple and swappable. It standardizes over the batch dimension only.
    Input is expected to be (N, D).

    必须用 register_buffer 注册 mean/var/initialized，否则不会写入 state_dict，eval 时
    会沿用初始值（如 var=1）或被异常/ padding 污染导致 eval 时除以错误方差、Loss 爆炸。
    训练时 normalize(..., update=True) 会更新这些 buffer；eval 时 update=False，只用已保存的统计量。
    """

    def __init__(self, dim: int, cfg: Optional[OnlineStandardizeConfig] = None):
        super().__init__()
        self.dim = int(dim)
        self.cfg = cfg or OnlineStandardizeConfig()

        self.register_buffer("mean", torch.zeros(self.dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(self.dim, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x shape (N,{self.dim}), got {tuple(x.shape)}")
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if not bool(self.initialized.item()):
            self.mean.copy_(batch_mean)
            self.var.copy_(torch.clamp_min(batch_var, self.cfg.clip_std_min**2))
            self.initialized.fill_(True)
            return

        m = float(self.cfg.momentum)
        self.mean.mul_(1.0 - m).add_(batch_mean, alpha=m)
        self.var.mul_(1.0 - m).add_(torch.clamp_min(batch_var, self.cfg.clip_std_min**2), alpha=m)

    def forward(self, x: torch.Tensor, *, update: bool) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected x shape (N,{self.dim}), got {tuple(x.shape)}")
        if update:
            self.update(x)
        # 安全锁：方差 < 1e-4 的维度视为常数/死通道，用 var=1 即 std=1，不做放大，避免 eval 时 (x-mean)/0.001 把微小噪声放大 1000 倍导致 Loss 爆炸（如 1139）
        var_safe_threshold = 1e-4
        safe_var = torch.where(
            self.var < var_safe_threshold,
            torch.ones_like(self.var, device=self.var.device, dtype=self.var.dtype),
            self.var,
        )
        std = torch.sqrt(safe_var + float(self.cfg.eps))
        return (x - self.mean) / std

