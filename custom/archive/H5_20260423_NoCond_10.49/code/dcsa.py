from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DCSAConfig:
    dim: int = 256
    num_heads: int = 8
    dropout: float = 0.0


class DiscreteContinuousSpatialAttention(nn.Module):
    """
    DCSA (Discrete-Continuous Spatial Attention).

    Minimal, pluggable implementation aligned with the "token-wise" fusion description:
      - Query from continuous tokens
      - Key/Value from discrete (codebook) tokens
      - Fusion: cont + Attn(cont, disc)

    Note:
      In the strict "i-to-i only" variant, attention reduces to a learned projection of the
      discrete token added as a residual. We keep MultiheadAttention for easy future extension
      (e.g., soft alignment / cross-token attention).
    """

    def __init__(self, cfg: DCSAConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.MultiheadAttention(
            embed_dim=int(cfg.dim),
            num_heads=int(cfg.num_heads),
            dropout=float(cfg.dropout),
            batch_first=True,
        )
        self.drop = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()
        self.norm = nn.LayerNorm(int(cfg.dim))

    def forward(self, cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
        """
        cont: (B, T, N, C)
        disc: (B, T, N, C)
        return: (B, T, N, C)
        """
        if cont.shape != disc.shape:
            raise ValueError(f"cont shape {tuple(cont.shape)} != disc shape {tuple(disc.shape)}")
        if cont.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), got {tuple(cont.shape)}")

        b, t, n, c = cont.shape
        xq = cont.reshape(b * t, n, c)
        xk = disc.reshape(b * t, n, c)

        # Token-wise alignment baseline: attend over tokens within the same frame.
        # If you want strict i-to-i, replace this with a diagonal-only mask.
        out, _ = self.attn(query=xq, key=xk, value=xk, need_weights=False)
        out = out.reshape(b, t, n, c)
        return self.norm(cont + self.drop(out))


class AsymmetricDCSA(nn.Module):
    """
    非对称 DCSA（论文 Section 3.2）：Query 来自 J 个连续关节点 Token，Key/Value 来自 N 个离散 Token。
    输出形状跟随 Query → (B, T, J, C)。J=25 保留关节点物理拓扑，N=63 为 codebook 积木。
    """

    def __init__(self, cfg: DCSAConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.MultiheadAttention(
            embed_dim=int(cfg.dim),
            num_heads=int(cfg.num_heads),
            dropout=float(cfg.dropout),
            batch_first=True,
        )
        self.drop = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()
        self.norm = nn.LayerNorm(int(cfg.dim))

    def forward(self, cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
        """
        cont: (B, T, Nq, C)  e.g. (B, T, 25, 256)  Query
        disc: (B, T, Nkv, C) e.g. (B, T, 63, 256) Key/Value
        return: (B, T, Nq, C)
        """
        if cont.ndim != 4 or disc.ndim != 4 or cont.shape[-1] != disc.shape[-1]:
            raise ValueError(f"Expected cont (B,T,Nq,C), disc (B,T,Nkv,C), C same; got {tuple(cont.shape)} {tuple(disc.shape)}")
        b, t, nq, c = cont.shape
        nkv = disc.shape[2]
        xq = cont.reshape(b * t, nq, c)
        xkv = disc.reshape(b * t, nkv, c)
        out, _ = self.attn(query=xq, key=xkv, value=xkv, need_weights=False)
        out = out.reshape(b, t, nq, c)
        return self.norm(cont + self.drop(out))

