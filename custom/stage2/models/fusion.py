"""
可插拔融合模块：离散 z_q（主干先验）+ 连续 z_e（细节）-> 融合特征 (B,T,N,C)。

- dcsa: Discrete-Continuous Spatial Attention（Query=连续, Key/Value=离散）
- residual_add: 残差相加 z_fused = z_q + detail_mlp(z_e)，轻量、梯度友好
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .dcsa import AsymmetricDCSA, DCSAConfig, DiscreteContinuousSpatialAttention


@dataclass(frozen=True)
class ResidualAddConfig:
    """残差相加融合：z_fused = z_disc + detail_mlp(z_cont)，防止连续特征主导。"""
    dim: int = 256
    # 细节 MLP：可选 0=仅缩放, 1=单层 Linear, 2=两层 MLP
    detail_layers: int = 2
    detail_hidden_ratio: float = 2.0  # hidden = dim * ratio
    # 若不使用 MLP，对 z_cont 的缩放因子（detail_layers=0 时有效）
    scale: float = 1.0
    dropout: float = 0.0


class ResidualAddFusion(nn.Module):
    """
    方法 1：残差相加。z_disc 为主干先验，z_cont 为细节残差。
    z_fused = z_disc + detail_mlp(z_cont)，零/轻量参数，梯度流好。
    """

    def __init__(self, cfg: ResidualAddConfig):
        super().__init__()
        self.cfg = cfg
        dim = int(cfg.dim)
        if int(cfg.detail_layers) == 0:
            self.detail_mlp = nn.Identity()
            self.scale = nn.Parameter(torch.tensor(float(cfg.scale)))
        elif int(cfg.detail_layers) == 1:
            self.detail_mlp = nn.Linear(dim, dim)
            self.scale = None
        else:
            hidden = max(dim, int(dim * float(cfg.detail_hidden_ratio)))
            self.detail_mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity(),
                nn.Linear(hidden, dim),
            )
            self.scale = None

    def forward(self, cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
        """
        cont: (B, T, N, C), disc: (B, T, N, C) -> (B, T, N, C)
        """
        if cont.shape != disc.shape or cont.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), cont {tuple(cont.shape)} disc {tuple(disc.shape)}")
        residual = self.detail_mlp(cont)
        if self.scale is not None:
            residual = residual * self.scale
        return disc + residual


def build_fusion(
    fusion_type: str,
    dim: int,
    *,
    dcsa_cfg: Optional[DCSAConfig] = None,
    residual_add_cfg: Optional[ResidualAddConfig] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    构建融合模块，接口统一：forward(cont, disc) -> (B,T,N,C)。
    """
    fusion_type = str(fusion_type).lower().strip()
    if fusion_type == "dcsa":
        cfg = dcsa_cfg or DCSAConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("num_heads", "dropout")})
        return DiscreteContinuousSpatialAttention(cfg)
    if fusion_type in ("dcsa_asymmetric", "dcsa_asym", "asymmetric_dcsa"):
        cfg = dcsa_cfg or DCSAConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("num_heads", "dropout")})
        return AsymmetricDCSA(cfg)
    if fusion_type in ("dcsa_symmetric", "dcsa_sym", "symmetric_dcsa"):
        from .dcsa import SymmetricDCSA
        cfg = dcsa_cfg or DCSAConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("num_heads", "dropout")})
        return SymmetricDCSA(cfg)
    if fusion_type in ("residual_add", "residual", "add"):
        cfg = residual_add_cfg or ResidualAddConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("detail_layers", "detail_hidden_ratio", "scale", "dropout")})
        return ResidualAddFusion(cfg)
    raise ValueError(f"Unknown fusion_type={fusion_type!r}. Use 'dcsa' or 'residual_add'.")
