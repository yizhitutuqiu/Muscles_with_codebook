from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DSTFormerConfig:
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    num_layers: int = 4


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(dim * float(mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DSTFormerBlock(nn.Module):
    """
    Minimal DSTFormer-like block:
      - spatial attention over tokens within each frame
      - temporal attention over frames for each token
    """

    def __init__(self, cfg: DSTFormerConfig):
        super().__init__()
        dim = int(cfg.dim)
        self.s_norm = nn.LayerNorm(dim)
        self.s_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(cfg.num_heads),
            dropout=float(cfg.attn_dropout),
            batch_first=True,
        )
        self.s_ffn = _FeedForward(dim, cfg.mlp_ratio, cfg.dropout)
        self.s_ffn_norm = nn.LayerNorm(dim)

        self.t_norm = nn.LayerNorm(dim)
        self.t_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(cfg.num_heads),
            dropout=float(cfg.attn_dropout),
            batch_first=True,
        )
        self.t_ffn = _FeedForward(dim, cfg.mlp_ratio, cfg.dropout)
        self.t_ffn_norm = nn.LayerNorm(dim)

        self.drop = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,C)
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), got {tuple(x.shape)}")
        b, t, n, c = x.shape

        # Spatial: (B*T, N, C)
        xs = self.s_norm(x).reshape(b * t, n, c)
        ys, _ = self.s_attn(query=xs, key=xs, value=xs, need_weights=False)
        ys = ys.reshape(b, t, n, c)
        x = x + self.drop(ys)
        x = x + self.s_ffn(self.s_ffn_norm(x))

        # Temporal: (B*N, T, C)
        xt = self.t_norm(x).transpose(1, 2).reshape(b * n, t, c)
        yt, _ = self.t_attn(query=xt, key=xt, value=xt, need_weights=False)
        yt = yt.reshape(b, n, t, c).transpose(1, 2)
        x = x + self.drop(yt)
        x = x + self.t_ffn(self.t_ffn_norm(x))
        return x


class DSTFormer(nn.Module):
    def __init__(self, cfg: DSTFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_DSTFormerBlock(cfg) for _ in range(int(cfg.num_layers))])
        self.out_norm = nn.LayerNorm(int(cfg.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)

