"""
EMG 头：将时序模块输出的 (B,T,N,C) 融合为 (B,T,8)，保留 N 个 token 的结构信息。

- mixer: 与 Stage1 一致，用 MLP-Mixer 对 N 个 token 做 token/channel mixing 后 Linear 到 8 维
- flatten: 展平 (B,T,N*C) 后两层 Linear，轻量
- spatial_pool: 论文对齐，在关节点维度 J 上 Average Pooling 得 (B,T,C)，再单层 FC 回归 8 维
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...models.mlp import MLPMixerConfig, FourLayerMLPMixerDecoder


@dataclass(frozen=True)
class EMGHeadConfig:
    token_count: int = 63
    dim: int = 256
    # mixer 专用
    mixer_hidden_dim: int = 256
    mixer_num_layers: int = 4
    # flatten 专用
    flatten_hidden_dim: int = 256


class MixerEMGHead(nn.Module):
    """
    与 Stage1 Decoder 一致：对每帧的 N 个 token 做 MLP-Mixer 融合，再 Linear 到 8 维。
    输入 (B,T,N,C)，输出 (B,T,8)。
    """

    def __init__(self, cfg: EMGHeadConfig):
        super().__init__()
        self.cfg = cfg
        n, c = int(cfg.token_count), int(cfg.dim)
        # FourLayerMLPMixerDecoder 的 in_dim 在 config 里表示 decoder 输出维（8）
        mixer_cfg = MLPMixerConfig(
            in_dim=8,
            token_count=n,
            code_dim=c,
            hidden_dim=int(cfg.mixer_hidden_dim),
            num_layers=int(cfg.mixer_num_layers),
        )
        self.decoder = FourLayerMLPMixerDecoder(mixer_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> (B, T, 8)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), got {tuple(x.shape)}")
        b, t, n, c = x.shape
        flat = x.reshape(b * t, n * c)
        out = self.decoder(flat)
        return out.view(b, t, 8)


class SpatialPoolingEMGHead(nn.Module):
    """
    论文严格对齐：Spatial Pooling (在关节点维度 J 上 mean) + 单层 FC。
    F_pool = Average Pooling(F_pose)，再通过 FC 回归 8 维 EMG。
    输入 (B,T,N,C)，输出 (B,T,8)。
    """

    def __init__(self, cfg: EMGHeadConfig):
        super().__init__()
        self.cfg = cfg
        c = int(cfg.dim)
        self.fc = nn.Linear(c, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> mean(dim=2) -> (B, T, C) -> Linear -> (B, T, 8)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), got {tuple(x.shape)}")
        pooled = x.mean(dim=2)
        return self.fc(pooled)


class FlattenEMGHead(nn.Module):
    """
    展平法：(B,T,N,C) -> (B,T,N*C) -> Linear -> GELU -> Linear -> (B,T,8)。
    """

    def __init__(self, cfg: EMGHeadConfig):
        super().__init__()
        self.cfg = cfg
        n, c = int(cfg.token_count), int(cfg.dim)
        hidden = int(cfg.flatten_hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(n * c),
            nn.Linear(n * c, hidden),
            nn.GELU(),
            nn.Linear(hidden, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> (B, T, 8)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,N,C), got {tuple(x.shape)}")
        b, t, n, c = x.shape
        flat = x.reshape(b, t, n * c)
        return self.net(flat)


# residual 模式下最后一层初始化：权重用极小高斯，偏置为 0，既保证初值≈0 又让梯度能回传（避免零初始化梯度断崖）
NEAR_ZERO_INIT_STD: float = 1e-4


def zero_init_final_output(emg_head: nn.Module) -> None:
    """
    对输出 8 维 EMG 的最后一层 Linear 做「近零初始化」：weight ~ N(0, 1e-4)，bias=0。
    用于 emg_pred_mode=residual 时：初值接近 0 不破坏 Stage1 保底，同时梯度可流回骨干，
    避免纯 zeros_(weight) 导致的梯度断崖（dL/dx = W^T dL/dy，W=0 时骨干收不到梯度）。
    """
    if hasattr(emg_head, "decoder") and hasattr(emg_head.decoder, "output_proj"):
        m = emg_head.decoder.output_proj
        if isinstance(m, nn.Linear) and m.out_features == 8:
            nn.init.normal_(m.weight, mean=0.0, std=NEAR_ZERO_INIT_STD)
            nn.init.zeros_(m.bias)
            return
    if hasattr(emg_head, "net") and isinstance(emg_head.net, nn.Sequential):
        for i in range(len(emg_head.net) - 1, -1, -1):
            m = emg_head.net[i]
            if isinstance(m, nn.Linear) and m.out_features == 8:
                nn.init.normal_(m.weight, mean=0.0, std=NEAR_ZERO_INIT_STD)
                nn.init.zeros_(m.bias)
                return
    raise RuntimeError("zero_init_final_output: could not find final Linear with out_features=8 in EMG head.")


def build_emg_head(head_type: str, cfg: EMGHeadConfig) -> nn.Module:
    """
    构建 EMG 头，接口统一：forward(x: (B,T,N,C)) -> (B,T,8)。
    """
    head_type = str(head_type).lower().strip()
    if head_type == "mixer":
        return MixerEMGHead(cfg)
    if head_type in ("flatten", "linear"):
        return FlattenEMGHead(cfg)
    if head_type in ("spatial_pool", "pool"):
        return SpatialPoolingEMGHead(cfg)
    raise ValueError(f"Unknown emg_head_type={head_type!r}. Use 'mixer', 'flatten', or 'spatial_pool'.")
