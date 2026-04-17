from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .mlp import MLPMixerConfig


@dataclass(frozen=True)
class TemporalConfig:
    """
    Temporal encoder/decoder config for clip-wise codebook.

    The model API stays (N, in_dim) <-> (N, token_count * code_dim) to keep the
    Stage1 codebook training loop minimally invasive. Internally we reshape:
      (N, seq_len * frame_dim) <-> (N, frame_dim, seq_len).
    """

    seq_len: Optional[int] = None
    frame_dim: Optional[int] = None

    kernel_size: int = 3
    num_layers: int = 4
    dilation_base: int = 2
    dropout: float = 0.0

    pool: str = "adaptive_avg"  # adaptive_avg | none | frame_tokens
    upsample: str = "linear"  # linear | none (frame-aligned: token_count == seq_len, no upsample)
    tokens_per_frame: Optional[int] = None


def _infer_seq_and_frame_dim(*, in_dim: int, temporal: TemporalConfig) -> tuple[int, int]:
    in_dim = int(in_dim)
    seq_len: Optional[int] = int(temporal.seq_len) if temporal.seq_len is not None else None
    frame_dim: Optional[int] = int(temporal.frame_dim) if temporal.frame_dim is not None else None

    if seq_len is not None and seq_len <= 0:
        raise ValueError(f"Invalid temporal.seq_len={seq_len}")

    if seq_len is None and frame_dim is None:
        if in_dim == 2250:
            seq_len, frame_dim = 30, 75
        elif in_dim == 240:
            seq_len, frame_dim = 30, 8
        else:
            raise ValueError(
                "temporal.seq_len is required for conv1d/tcn encoder/decoder when in_dim is not 2250 or 240; "
                "set temporal.seq_len (and optionally temporal.frame_dim) in config."
            )
    elif seq_len is None and frame_dim is not None:
        if frame_dim <= 0 or in_dim % frame_dim != 0:
            raise ValueError(f"Cannot infer seq_len: in_dim={in_dim} not divisible by frame_dim={frame_dim}")
        seq_len = in_dim // frame_dim
    elif seq_len is not None and frame_dim is None:
        if in_dim % seq_len != 0:
            raise ValueError(f"in_dim={in_dim} not divisible by seq_len={seq_len}; set temporal.frame_dim explicitly")
        frame_dim = in_dim // seq_len
    else:
        if seq_len * frame_dim != in_dim:
            raise ValueError(f"in_dim={in_dim} != seq_len*frame_dim ({seq_len}*{frame_dim})")
    return int(seq_len), int(frame_dim)


class _TemporalBlock(nn.Module):
    """
    TCN 残差块。使用 GroupNorm 替代 BatchNorm1d，避免 running stats 在 eval 时损坏导致 Loss 爆炸。
    GroupNorm(channels, channels) 对 (N,C,L) 按通道归一化，不依赖历史统计量。
    """

    def __init__(self, *, channels: int, kernel_size: int, dilation: int, dropout: float, use_residual: bool):
        super().__init__()
        pad = (int(kernel_size) - 1) * int(dilation) // 2
        self.conv = nn.Conv1d(
            int(channels),
            int(channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=int(pad),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        c = int(channels)
        # 优化性能：如果 num_groups=c (相当于 InstanceNorm1d)，在 L=5 等极短序列上 GPU 吞吐量极差。
        # 改为固定 num_groups=32 (或 c 若 c<32)，可将速度提升 10-20 倍。
        # 注意：这会改变数值行为，若已有 checkpoint 需重新训练。
        ng = 32 if c >= 32 and c % 32 == 0 else (1 if c < 32 else 8)
        self.norm = nn.GroupNorm(num_groups=ng, num_channels=c)
        self.use_residual = bool(use_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.drop(y)
        return (x + y) if self.use_residual else y


class TemporalConv1dEncoder(nn.Module):
    """
    Clip-wise encoder: (N, seq_len*frame_dim) -> (N, token_count*code_dim).
    Uses conv1d along time (seq_len), then adaptive pooling to token_count.
    无跨样本/跨 batch 隐状态，每次 forward 处理 N 个独立片段，无需 reset。
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        token_count: int,
        code_dim: int,
        temporal: TemporalConfig,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_count = int(token_count)
        self.code_dim = int(code_dim)
        self.temporal = temporal

        seq_len, frame_dim = _infer_seq_and_frame_dim(in_dim=self.in_dim, temporal=temporal)
        self.seq_len = int(seq_len)
        self.frame_dim = int(frame_dim)

        k = int(temporal.kernel_size)
        layers = int(temporal.num_layers)
        drop = float(temporal.dropout)

        self.in_proj = nn.Conv1d(self.frame_dim, self.hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, layers)):
            blocks.append(_TemporalBlock(channels=self.hidden_dim, kernel_size=k, dilation=1, dropout=drop, use_residual=False))
        self.blocks = nn.Sequential(*blocks)
        self.to_code = nn.Conv1d(self.hidden_dim, self.code_dim, kernel_size=1)
        pool_mode = str(self.temporal.pool).lower().strip()
        if pool_mode == "frame_tokens":
            tpf = getattr(self.temporal, "tokens_per_frame", None)
            if tpf is None:
                raise ValueError("temporal.pool=frame_tokens requires temporal.tokens_per_frame")
            self._tokens_per_frame = int(tpf)
            if self._tokens_per_frame <= 0:
                raise ValueError(f"Invalid temporal.tokens_per_frame={self._tokens_per_frame}")
            if int(self.token_count) != int(self.seq_len) * int(self._tokens_per_frame):
                raise ValueError(
                    f"temporal.pool=frame_tokens requires token_count=seq_len*tokens_per_frame, "
                    f"got token_count={self.token_count}, seq_len={self.seq_len}, tokens_per_frame={self._tokens_per_frame}"
                )
            self.expand = nn.Conv1d(self.code_dim, self.code_dim * self._tokens_per_frame, kernel_size=1)
        else:
            self._tokens_per_frame = None
            self.expand = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape (N,{self.in_dim}), got {tuple(x.shape)}")
        n = int(x.shape[0])
        xt = x.view(n, self.seq_len, self.frame_dim).transpose(1, 2).contiguous()  # (N,C,L)
        y = self.in_proj(xt)
        y = self.blocks(y)
        y = self.to_code(y)  # (N, code_dim, L)

        pool_mode = str(self.temporal.pool).lower().strip()
        if pool_mode == "adaptive_avg":
            y = F.adaptive_avg_pool1d(y, output_size=int(self.token_count))  # (N, code_dim, token_count)
        elif pool_mode == "none":
            if int(self.token_count) != self.seq_len:
                raise ValueError(
                    f"temporal.pool=none (frame-aligned) requires token_count={self.token_count} == seq_len={self.seq_len}"
                )
            # 30-in-30-out: no pool, keep seq_len time steps
        elif pool_mode == "frame_tokens":
            if self.expand is None or self._tokens_per_frame is None:
                raise RuntimeError("frame_tokens mode requires expand conv; unexpected None")
            y = self.expand(y)  # (N, code_dim*tokens_per_frame, seq_len)
            y = (
                y.view(n, int(self._tokens_per_frame), int(self.code_dim), int(self.seq_len))
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # (N, seq_len, tokens_per_frame, code_dim)
            y = y.view(n, int(self.token_count), int(self.code_dim))  # (N, token_count, code_dim)
            return y.view(n, int(self.token_count) * int(self.code_dim))
        else:
            raise ValueError(f"Unsupported temporal.pool: {self.temporal.pool}")
        y = y.transpose(1, 2).contiguous()  # (N, token_count, code_dim)
        return y.view(n, int(self.token_count) * int(self.code_dim))


class TemporalTCNEncoder(nn.Module):
    """
    TCN-style encoder with residual dilated conv blocks.
    无跨样本/跨 batch 隐状态（纯 Conv1d+GroupNorm），每次 forward 处理 N 个独立 30 帧片段，无需 reset。
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        token_count: int,
        code_dim: int,
        temporal: TemporalConfig,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_count = int(token_count)
        self.code_dim = int(code_dim)
        self.temporal = temporal

        seq_len, frame_dim = _infer_seq_and_frame_dim(in_dim=self.in_dim, temporal=temporal)
        self.seq_len = int(seq_len)
        self.frame_dim = int(frame_dim)

        k = int(temporal.kernel_size)
        layers = int(temporal.num_layers)
        dilation_base = int(temporal.dilation_base)
        drop = float(temporal.dropout)

        self.in_proj = nn.Conv1d(self.frame_dim, self.hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, layers)):
            dilation = int(dilation_base) ** int(i) if dilation_base > 1 else 1
            blocks.append(_TemporalBlock(channels=self.hidden_dim, kernel_size=k, dilation=dilation, dropout=drop, use_residual=True))
        self.blocks = nn.Sequential(*blocks)
        self.to_code = nn.Conv1d(self.hidden_dim, self.code_dim, kernel_size=1)
        pool_mode = str(self.temporal.pool).lower().strip()
        if pool_mode == "frame_tokens":
            tpf = getattr(self.temporal, "tokens_per_frame", None)
            if tpf is None:
                raise ValueError("temporal.pool=frame_tokens requires temporal.tokens_per_frame")
            self._tokens_per_frame = int(tpf)
            if self._tokens_per_frame <= 0:
                raise ValueError(f"Invalid temporal.tokens_per_frame={self._tokens_per_frame}")
            if int(self.token_count) != int(self.seq_len) * int(self._tokens_per_frame):
                raise ValueError(
                    f"temporal.pool=frame_tokens requires token_count=seq_len*tokens_per_frame, "
                    f"got token_count={self.token_count}, seq_len={self.seq_len}, tokens_per_frame={self._tokens_per_frame}"
                )
            self.expand = nn.Conv1d(self.code_dim, self.code_dim * self._tokens_per_frame, kernel_size=1)
        else:
            self._tokens_per_frame = None
            self.expand = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape (N,{self.in_dim}), got {tuple(x.shape)}")
        n = int(x.shape[0])
        xt = x.view(n, self.seq_len, self.frame_dim).transpose(1, 2).contiguous()  # (N,C,L)
        y = self.in_proj(xt)
        y = self.blocks(y)
        y = self.to_code(y)  # (N, code_dim, L)

        pool_mode = str(self.temporal.pool).lower().strip()
        if pool_mode == "adaptive_avg":
            y = F.adaptive_avg_pool1d(y, output_size=int(self.token_count))
        elif pool_mode == "none":
            if int(self.token_count) != self.seq_len:
                raise ValueError(
                    f"temporal.pool=none (frame-aligned) requires token_count={self.token_count} == seq_len={self.seq_len}"
                )
            # 30-in-30-out: no pool, keep seq_len time steps
        elif pool_mode == "frame_tokens":
            if self.expand is None or self._tokens_per_frame is None:
                raise RuntimeError("frame_tokens mode requires expand conv; unexpected None")
            y = self.expand(y)  # (N, code_dim*tokens_per_frame, seq_len)
            y = (
                y.view(n, int(self._tokens_per_frame), int(self.code_dim), int(self.seq_len))
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # (N, seq_len, tokens_per_frame, code_dim)
            y = y.view(n, int(self.token_count), int(self.code_dim))  # (N, token_count, code_dim)
            return y.view(n, int(self.token_count) * int(self.code_dim))
        else:
            raise ValueError(f"Unsupported temporal.pool: {self.temporal.pool}")
        y = y.transpose(1, 2).contiguous()
        return y.view(n, int(self.token_count) * int(self.code_dim))


class MixerTCNEncoder(nn.Module):
    """
    Encoder: input -> Mixer (token mixing) -> TCN (temporal conv). Used when encoder_type is mixer_tcn.
    Mixer maps (N, in_dim) -> (N, mixer_token_count * mixer_code_dim); TCN then treats that as
    (N, seq_len, frame_dim) and does temporal encoding -> (N, token_count * code_dim).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        mixer_cfg: "MLPMixerConfig",
        tcn_hidden_dim: int,
        token_count: int,
        code_dim: int,
        temporal: TemporalConfig,
    ):
        super().__init__()
        from .mlp import FourLayerMLPMixer

        self.in_dim = int(in_dim)
        self.token_count = int(token_count)
        self.code_dim = int(code_dim)
        self.mixer = FourLayerMLPMixer(mixer_cfg)
        mixer_out_dim = int(mixer_cfg.token_count) * int(mixer_cfg.code_dim)
        self.tcn = TemporalTCNEncoder(
            in_dim=mixer_out_dim,
            hidden_dim=int(tcn_hidden_dim),
            token_count=int(token_count),
            code_dim=int(code_dim),
            temporal=temporal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape (N,{self.in_dim}), got {tuple(x.shape)}")
        z = self.mixer(x)
        return self.tcn(z)


class TemporalConvDecoder(nn.Module):
    """
    Clip-wise decoder: (N, token_count*code_dim) -> (N, seq_len*frame_dim).

    Upsamples tokens back to seq_len, then projects per-timestep to frame_dim.
    """

    def __init__(
        self,
        *,
        out_dim: int,
        token_count: int,
        code_dim: int,
        temporal: TemporalConfig,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.token_count = int(token_count)
        self.code_dim = int(code_dim)
        self.temporal = temporal

        seq_len, frame_dim = _infer_seq_and_frame_dim(in_dim=self.out_dim, temporal=temporal)
        self.seq_len = int(seq_len)
        self.frame_dim = int(frame_dim)

        self.to_frame = nn.Conv1d(self.code_dim, self.frame_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != int(self.token_count) * int(self.code_dim):
            raise ValueError(f"Expected z shape (N,{self.token_count*self.code_dim}), got {tuple(z.shape)}")
        n = int(z.shape[0])
        y = z.view(n, int(self.token_count), int(self.code_dim)).transpose(1, 2).contiguous()  # (N, code_dim, token_count)

        up = str(self.temporal.upsample).lower().strip()
        if up == "linear":
            y = F.interpolate(y, size=int(self.seq_len), mode="linear", align_corners=False)  # (N, code_dim, seq_len)
        elif up == "none":
            if int(self.token_count) != self.seq_len:
                raise ValueError(
                    f"temporal.upsample=none (frame-aligned) requires token_count={self.token_count} == seq_len={self.seq_len}"
                )
            # 30-in-30-out: no upsample, token_count == seq_len
        else:
            raise ValueError(f"Unsupported temporal.upsample: {self.temporal.upsample}")
        y = self.to_frame(y)  # (N, frame_dim, seq_len)
        y = y.transpose(1, 2).contiguous().view(n, int(self.seq_len) * int(self.frame_dim))
        return y


class ClipUnifiedTCNEncoder(nn.Module):
    """
    ClipUnifiedTCNEncoder: (N, seq_len*frame_dim) -> (N, code_dim).
    Encodes the whole clip into a single token (token_count=1).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        code_dim: int,
        temporal: TemporalConfig,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.temporal = temporal

        seq_len, frame_dim = _infer_seq_and_frame_dim(in_dim=self.in_dim, temporal=temporal)
        self.seq_len = int(seq_len)
        self.frame_dim = int(frame_dim)

        k = int(temporal.kernel_size)
        layers = int(temporal.num_layers)
        dilation_base = int(temporal.dilation_base)
        drop = float(temporal.dropout)

        self.in_proj = nn.Conv1d(self.frame_dim, self.hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, layers)):
            dilation = int(dilation_base) ** int(i) if dilation_base > 1 else 1
            blocks.append(_TemporalBlock(channels=self.hidden_dim, kernel_size=k, dilation=dilation, dropout=drop, use_residual=True))
        self.blocks = nn.Sequential(*blocks)
        self.to_code = nn.Conv1d(self.hidden_dim, self.code_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Expected x shape (N,{self.in_dim}), got {tuple(x.shape)}")
        n = int(x.shape[0])
        # Reshape to (N, F, T)
        xt = x.view(n, self.seq_len, self.frame_dim).transpose(1, 2).contiguous()
        
        y = self.in_proj(xt)
        y = self.blocks(y)
        
        # Global temporal pooling -> (N, hidden_dim, 1)
        y = F.adaptive_avg_pool1d(y, output_size=1)
        
        # Project to code_dim -> (N, code_dim, 1)
        y = self.to_code(y)
        
        # Flatten to (N, code_dim)
        return y.view(n, self.code_dim)


class ClipUnifiedTCNDecoder(nn.Module):
    """
    ClipUnifiedTCNDecoder: (N, code_dim) -> (N, seq_len*frame_dim).
    Decodes a single unified token back into the whole clip.
    """

    def __init__(
        self,
        *,
        out_dim: int,
        hidden_dim: int,
        code_dim: int,
        temporal: TemporalConfig,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.code_dim = int(code_dim)
        self.temporal = temporal

        seq_len, frame_dim = _infer_seq_and_frame_dim(in_dim=self.out_dim, temporal=temporal)
        self.seq_len = int(seq_len)
        self.frame_dim = int(frame_dim)

        # Positional encoding for the T clip frames
        self.pos_embed = nn.Parameter(torch.zeros(1, self.code_dim, self.seq_len))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        k = int(temporal.kernel_size)
        layers = int(temporal.num_layers)
        dilation_base = int(temporal.dilation_base)
        drop = float(temporal.dropout)

        self.in_proj = nn.Conv1d(self.code_dim, self.hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, layers)):
            dilation = int(dilation_base) ** int(i) if dilation_base > 1 else 1
            blocks.append(_TemporalBlock(channels=self.hidden_dim, kernel_size=k, dilation=dilation, dropout=drop, use_residual=True))
        self.blocks = nn.Sequential(*blocks)
        self.to_frame = nn.Conv1d(self.hidden_dim, self.frame_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.code_dim:
            raise ValueError(f"Expected z shape (N,{self.code_dim}), got {tuple(z.shape)}")
        n = int(z.shape[0])
        
        # Expand code to (N, code_dim, T)
        y = z.view(n, self.code_dim, 1).repeat(1, 1, self.seq_len)
        
        # Add positional encoding
        y = y + self.pos_embed
        
        y = self.in_proj(y)
        y = self.blocks(y)
        y = self.to_frame(y)  # (N, frame_dim, T)
        
        # Reshape back to (N, T * F)
        y = y.transpose(1, 2).contiguous().view(n, self.seq_len * self.frame_dim)
        return y

