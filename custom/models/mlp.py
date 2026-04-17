from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MLPConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    act: str = "silu"
    dropout: float = 0.0


@dataclass(frozen=True)
class MLPMixerConfig:
    in_dim: int
    token_count: int
    code_dim: int
    hidden_dim: int
    num_layers: int = 4
    token_mlp_dim: Optional[int] = None
    channel_mlp_dim: Optional[int] = None
    act: str = "gelu"
    dropout: float = 0.0


def _make_act(name: str) -> nn.Module:
    name = str(name).lower().strip()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class TwoLayerMLP(nn.Module):
    """Minimal 2-layer MLP block used for both encoder and decoder."""

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            _make_act(cfg.act),
            nn.Dropout(p=float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity(),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MixerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, *, act: str, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            _make_act(act),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MLPMixerBlock(nn.Module):
    def __init__(self, cfg: MLPMixerConfig):
        super().__init__()
        token_mlp_dim = int(cfg.token_mlp_dim or cfg.hidden_dim)
        channel_mlp_dim = int(cfg.channel_mlp_dim or cfg.hidden_dim)
        self.token_norm = nn.LayerNorm(int(cfg.code_dim))
        self.token_mlp = _MixerMLP(
            int(cfg.token_count),
            token_mlp_dim,
            act=cfg.act,
            dropout=float(cfg.dropout),
        )
        self.channel_norm = nn.LayerNorm(int(cfg.code_dim))
        self.channel_mlp = _MixerMLP(
            int(cfg.code_dim),
            channel_mlp_dim,
            act=cfg.act,
            dropout=float(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class FourLayerMLPMixer(nn.Module):
    """
    Project frame features to token features, then refine them with 4 MLP-Mixer blocks.

    Input:  (B, in_dim)
    Output: (B, token_count * code_dim)
    """

    def __init__(self, cfg: MLPMixerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_count = int(cfg.token_count)
        self.code_dim = int(cfg.code_dim)
        out_dim = self.token_count * self.code_dim
        self.input_proj = nn.Linear(int(cfg.in_dim), out_dim)
        self.blocks = nn.ModuleList([_MLPMixerBlock(cfg) for _ in range(int(cfg.num_layers))])
        self.output_norm = nn.LayerNorm(self.code_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != int(self.cfg.in_dim):
            raise ValueError(f"Expected x shape (B,{self.cfg.in_dim}), got {tuple(x.shape)}")
        x = self.input_proj(x).view(int(x.shape[0]), self.token_count, self.code_dim)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return x.reshape(int(x.shape[0]), self.token_count * self.code_dim)


class FourLayerMLPMixerDecoder(nn.Module):
    """
    Symmetric MLP-Mixer decoder:
      (B, token_count * code_dim) -> (B, token_count, code_dim) -> 4 mixer blocks
      -> flatten -> Linear -> (B, out_dim)
    """

    def __init__(self, cfg: MLPMixerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_count = int(cfg.token_count)
        self.code_dim = int(cfg.code_dim)
        in_dim = self.token_count * self.code_dim
        self.input_norm = nn.LayerNorm(self.code_dim)
        self.blocks = nn.ModuleList([_MLPMixerBlock(cfg) for _ in range(int(cfg.num_layers))])
        self.output_proj = nn.Linear(in_dim, int(cfg.in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_dim = self.token_count * self.code_dim
        if x.ndim != 2 or x.shape[1] != expected_dim:
            raise ValueError(f"Expected x shape (B,{expected_dim}), got {tuple(x.shape)}")
        x = x.view(int(x.shape[0]), self.token_count, self.code_dim)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x.reshape(int(x.shape[0]), expected_dim))

