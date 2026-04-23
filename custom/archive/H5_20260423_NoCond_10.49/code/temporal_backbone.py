"""
可插拔融合后时序模块：融合特征 (B,T,N,C) -> 时序提炼后仍保留 N 的 (B,T,N,C)。

不再在时序出口做 mean(dim=2)，以保留 63 个 token 的空间/结构信息，供 EMG 头做 Mixer/展平融合。
- dstformer: DSTFormer 时空注意力，直接输出 (B,T,N,C)
- tcn: 对每个 token 沿时间维 TCN，输出 (B,T,N,C)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from .dstformer import DSTFormer, DSTFormerConfig
from .dstformer_v2 import DSTFormerV2, DSTFormerV2Config


@dataclass(frozen=True)
class TCNBackboneConfig:
    """融合后 TCN：沿时间维 1D 卷积，输入 (B,T,C)，输出 (B,T,C)。"""
    dim: int = 256
    hidden_dim: int = 256
    kernel_size: int = 3
    num_layers: int = 4
    dilation_base: int = 2
    dropout: float = 0.0


class _TCNBlock(nn.Module):
    """TCN 残差块，GroupNorm，无跨 batch 隐状态。"""

    def __init__(self, *, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (int(kernel_size) - 1) * int(dilation) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.norm = nn.GroupNorm(num_groups=channels, num_channels=channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.norm(self.conv(x))))


class TCNTemporalBackbone(nn.Module):
    """
    融合后 TCN：对每个 token 沿时间维做 1D 卷积，保留 N 维。输入 (B,T,N,C) -> 输出 (B,T,N,C)。
    无跨样本/跨 batch 隐状态。
    """

    def __init__(self, cfg: TCNBackboneConfig):
        super().__init__()
        self.cfg = cfg
        dim = int(cfg.dim)
        hidden = int(cfg.hidden_dim)
        k = int(cfg.kernel_size)
        layers = int(cfg.num_layers)
        dilation_base = int(cfg.dilation_base)
        drop = float(cfg.dropout)

        self.in_proj = nn.Conv1d(dim, hidden, kernel_size=1)
        blocks = []
        for i in range(max(1, layers)):
            dilation = dilation_base ** i if dilation_base > 1 else 1
            blocks.append(_TCNBlock(channels=hidden, kernel_size=k, dilation=dilation, dropout=drop))
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(hidden, dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> 输出 (B, T, N, C)，保留 N 维供 EMG 头做 token 级融合
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), got {tuple(x.shape)}")
        b, t, n, c = x.shape
        # (B, N, T, C) -> (B*N, C, T)
        xt = x.transpose(1, 2).reshape(b * n, t, c).transpose(1, 2).contiguous()
        y = self.in_proj(xt)
        y = self.blocks(y)
        y = self.out_proj(y)
        y = y.transpose(1, 2).reshape(b, n, t, c).transpose(1, 2).contiguous()
        return self.out_norm(y)


class DSTFormerTemporalBackbone(nn.Module):
    """
    DSTFormer：输入 (B,T,N,C)，输出 (B,T,N,C)。不再做 mean(dim=2)，保留 N 供 EMG 头融合。
    """

    def __init__(self, cfg: DSTFormerConfig):
        super().__init__()
        self.dst = DSTFormer(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> 输出 (B, T, N, C)
        """
        return self.dst(x)


class DSTFormerV2TemporalBackbone(nn.Module):
    """
    DSTFormerV2 (H5)：包含 RoPE, LTC, SwiGLU 的增强型时序架构，提高无条件泛化能力。
    """

    def __init__(self, cfg: DSTFormerV2Config):
        super().__init__()
        self.dst = DSTFormerV2(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> 输出 (B, T, N, C)
        """
        return self.dst(x)


def build_temporal_backbone(
    temporal_type: str,
    dim: int,
    *,
    dst_cfg: Optional[DSTFormerConfig] = None,
    dst_v2: Optional[DSTFormerV2Config] = None,
    tcn_cfg: Optional[TCNBackboneConfig] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    构建融合后时序模块，接口统一：forward(x: (B,T,N,C)) -> (B,T,N,C)。不再做 mean，由 EMG 头负责 N 维融合。
    """
    temporal_type = str(temporal_type).lower().strip()
    if temporal_type in ("dstformer", "dst", "former"):
        cfg = dst_cfg or DSTFormerConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers")})
        return DSTFormerTemporalBackbone(cfg)
    if temporal_type in ("dstformer_v2", "dst_v2", "h5"):
        # Allow reusing dst_cfg fields for backward compatibility in config files
        cfg_dict = {"dim": dim}
        if dst_cfg is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers"):
                if hasattr(dst_cfg, k):
                    cfg_dict[k] = getattr(dst_cfg, k)
        if dst_v2 is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc"):
                if hasattr(dst_v2, k):
                    cfg_dict[k] = getattr(dst_v2, k)
        cfg_dict.update({k: v for k, v in kwargs.items() if k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc")})
        cfg = DSTFormerV2Config(**cfg_dict)
        return DSTFormerV2TemporalBackbone(cfg)
    if temporal_type == "tcn":
        cfg = tcn_cfg or TCNBackboneConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("hidden_dim", "kernel_size", "num_layers", "dilation_base", "dropout")})
        return TCNTemporalBackbone(cfg)
    raise ValueError(f"Unknown temporal_type={temporal_type!r}. Use 'dstformer' or 'tcn'.")
