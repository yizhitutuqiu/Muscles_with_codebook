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
import math

from .dcsa import DCSAConfig, SymmetricDCSA
from .dstformer import DSTFormer, DSTFormerConfig
from .dstformer_v2 import DSTFormerV2, DSTFormerV2Config
from .dstformer_v3_moe import DSTFormerV3MoE, DSTFormerV3MoEConfig
from .dstformer_v4_dual_moe import DSTFormerV4DualMoE, DSTFormerV4DualMoEConfig
from .dstformer_v5_guided_moe import DSTFormerV5GuidedMoE, DSTFormerV5GuidedMoEConfig


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


class DSTFormerV5GuidedMoETemporalBackbone(nn.Module):
    """
    Wrapper for DSTFormerV5GuidedMoE
    """
    def __init__(self, cfg: DSTFormerV5GuidedMoEConfig):
        super().__init__()
        self.dst = DSTFormerV5GuidedMoE(cfg)

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, N, C)
        # guide: (B, T, C) or None
        return self.dst(x, guide=guide)

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


class DSTFormerV3MoETemporalBackbone(nn.Module):
    """
    DSTFormerV3MoE (H6-C): Kinematic-driven Mixture of Experts to route different spatial/temporal parts 
    without relying on explicit condition.
    """
    def __init__(self, cfg: DSTFormerV3MoEConfig):
        super().__init__()
        self.dst = DSTFormerV3MoE(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> 输出 (B, T, N, C)
        """
        return self.dst(x)


class DSTFormerV4DualMoETemporalBackbone(nn.Module):
    """
    DSTFormerV4DualMoE: Decoupled Spatial-Temporal MoE routing.
    """
    def __init__(self, cfg: DSTFormerV4DualMoEConfig):
        super().__init__()
        self.dst = DSTFormerV4DualMoE(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, C) -> 输出 (B, T, N, C)
        """
        return self.dst(x)


class MiaOfficialLightTransformerTemporal(nn.Module):
    produces_pred: bool = True

    def __init__(self, *, task: str):
        super().__init__()
        self.task = str(task).strip().lower()
        if self.task == "pose2emg":
            from musclesinaction.models import modelposetoemg as _transmodel

            self.net = _transmodel.TransformerEnc(
                threed="True",
                num_tokens=50,
                dim_model=128,
                num_classes=20,
                num_heads=16,
                classif=False,
                num_encoder_layers=8,
                num_decoder_layers=3,
                dropout_p=0.1,
                device="cuda",
                embedding=True,
                step=30,
            )
            self._expected_in = "joints3d"
            self._out_dim = 8
        elif self.task == "emg2pose":
            from musclesinaction.models import modelemgtopose as _transmodel

            self.net = _transmodel.TransformerEnc(
                threed="True",
                num_tokens=50,
                dim_model=128,
                num_classes=20,
                num_heads=16,
                classif=False,
                num_encoder_layers=8,
                num_decoder_layers=3,
                dropout_p=0.1,
                device="cuda",
                embedding=True,
                step=30,
            )
            self._expected_in = "emg"
            self._out_dim = 75
        else:
            raise ValueError(f"Unknown task={task!r} for MiaOfficialLightTransformerTemporal")

        self._expected_param_count = sum(p.numel() for p in self.net.parameters())

    def forward(self, x: torch.Tensor, *, raw_inputs: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if raw_inputs.ndim not in (3, 4):
            raise ValueError(f"raw_inputs must be (B,T,8) or (B,T,25,3), got {tuple(raw_inputs.shape)}")
        b, t = int(raw_inputs.shape[0]), int(raw_inputs.shape[1])
        if t != 30:
            raise ValueError(f"Official transformer expects T=30 (step=30), got T={t}")

        if cond is None:
            condval = torch.zeros((b,), device=raw_inputs.device, dtype=torch.float32)
        else:
            condval = cond.reshape(b).to(device=raw_inputs.device, dtype=torch.float32)

        if x.ndim != 4:
            raise ValueError(f"Expected fused features x as (B,T,N,C), got {tuple(x.shape)}")
        fused_bias = x.mean(dim=2)
        if fused_bias.shape[-1] < 126:
            fused_bias = torch.nn.functional.pad(fused_bias, (0, 126 - fused_bias.shape[-1]))
        fused_bias = fused_bias[:, :, :126]

        if self._expected_in == "joints3d":
            if raw_inputs.ndim != 4 or raw_inputs.shape[-2:] != (25, 3):
                raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(raw_inputs.shape)}")
            src = raw_inputs.reshape(b, t, 75)
            src = src.float() * math.sqrt(self.net.dim_model)
            src = torch.unsqueeze(src, dim=1).permute(0, 1, 3, 2)
            srcorig = self.net.conv1(src)[:, :, 0, :].permute(0, 2, 1)
            srcorig = srcorig + fused_bias.to(device=srcorig.device, dtype=srcorig.dtype)
            src = self.net.positional_encoder(srcorig)
            condition = torch.ones(src.shape[0], src.shape[1], 2, device=src.device, dtype=src.dtype) * condval.reshape(
                condval.shape[0], 1, 1
            ).to(device=src.device, dtype=src.dtype)
            srccat = torch.cat([src, condition], dim=2)
            src = srccat.permute(1, 0, 2)
            transformer_out = self.net.transformer0(src, src_key_padding_mask=None)
            out0 = self.net.out0(transformer_out)
            out0 = out0.permute(1, 2, 0)
            return out0.permute(0, 2, 1)

        if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 8:
            raise ValueError(f"Expected emg (B,T,8), got {tuple(raw_inputs.shape)}")
        src = raw_inputs.permute(0, 2, 1)
        src = src.float() * math.sqrt(self.net.dim_model)
        src = torch.unsqueeze(src, dim=1).permute(0, 1, 2, 3)
        src = self.net.conv1(src)[:, :, 0, :].permute(0, 2, 1)
        src = src + fused_bias.to(device=src.device, dtype=src.dtype)
        src = self.net.positional_encoder(src)
        condition = torch.ones(src.shape[0], src.shape[1], 2, device=src.device, dtype=src.dtype) * condval.reshape(
            condval.shape[0], 1, 1
        ).to(device=src.device, dtype=src.dtype)
        srccat = torch.cat([src, condition], dim=2)
        src = srccat.permute(1, 0, 2)
        transformer_out = self.net.transformer0(src, src_key_padding_mask=None)
        out0 = self.net.out0(transformer_out)
        out0 = out0.permute(1, 2, 0)
        return out0.permute(0, 2, 1)


class MiaOfficialLightTransformerConvDCSATemporal(nn.Module):
    produces_pred: bool = True
    expects_disc_tokens: bool = True

    def __init__(self, *, task: str, dcsa_cfg: DCSAConfig):
        super().__init__()
        self.task = str(task).strip().lower()
        self.dcsa = SymmetricDCSA(dcsa_cfg)
        self._conv_dim = 126
        d_model = int(dcsa_cfg.dim)
        if d_model != int(self._conv_dim):
            self.to_dcsa = nn.Linear(int(self._conv_dim), d_model)
            self.from_dcsa = nn.Linear(d_model, int(self._conv_dim))
        else:
            self.to_dcsa = nn.Identity()
            self.from_dcsa = nn.Identity()
        if self.task == "pose2emg":
            from musclesinaction.models import modelposetoemg as _transmodel

            self.net = _transmodel.TransformerEnc(
                threed="True",
                num_tokens=50,
                dim_model=128,
                num_classes=20,
                num_heads=16,
                classif=False,
                num_encoder_layers=8,
                num_decoder_layers=3,
                dropout_p=0.1,
                device="cuda",
                embedding=True,
                step=30,
            )
            self._expected_in = "joints3d"
            self._out_dim = 8
        elif self.task == "emg2pose":
            from musclesinaction.models import modelemgtopose as _transmodel

            self.net = _transmodel.TransformerEnc(
                threed="True",
                num_tokens=50,
                dim_model=128,
                num_classes=20,
                num_heads=16,
                classif=False,
                num_encoder_layers=8,
                num_decoder_layers=3,
                dropout_p=0.1,
                device="cuda",
                embedding=True,
                step=30,
            )
            self._expected_in = "emg"
            self._out_dim = 75
        else:
            raise ValueError(f"Unknown task={task!r} for MiaOfficialLightTransformerConvDCSATemporal")

        self._expected_param_count = sum(p.numel() for p in self.net.parameters())

    def forward(self, x: torch.Tensor, *, raw_inputs: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if raw_inputs.ndim not in (3, 4):
            raise ValueError(f"raw_inputs must be (B,T,8) or (B,T,25,3), got {tuple(raw_inputs.shape)}")
        b, t = int(raw_inputs.shape[0]), int(raw_inputs.shape[1])
        if t != 30:
            raise ValueError(f"Official transformer expects T=30 (step=30), got T={t}")

        if cond is None:
            condval = torch.zeros((b,), device=raw_inputs.device, dtype=torch.float32)
        else:
            condval = cond.reshape(b).to(device=raw_inputs.device, dtype=torch.float32)

        if x.ndim != 4:
            raise ValueError(f"Expected discrete tokens x as (B,T,N,C), got {tuple(x.shape)}")

        if self._expected_in == "joints3d":
            if raw_inputs.ndim != 4 or raw_inputs.shape[-2:] != (25, 3):
                raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(raw_inputs.shape)}")
            src = raw_inputs.reshape(b, t, 75)
            src = src.float() * math.sqrt(self.net.dim_model)
            src = torch.unsqueeze(src, dim=1).permute(0, 1, 3, 2)
            srcorig = self.net.conv1(src)[:, :, 0, :].permute(0, 2, 1)
            z_cont = self.to_dcsa(srcorig).unsqueeze(2)
            z_fused = self.dcsa(z_cont, x)
            src_fused = self.from_dcsa(z_fused[:, :, : z_cont.shape[2], :].mean(dim=2))
            src = self.net.positional_encoder(src_fused)
            condition = torch.ones(src.shape[0], src.shape[1], 2, device=src.device, dtype=src.dtype) * condval.reshape(
                condval.shape[0], 1, 1
            ).to(device=src.device, dtype=src.dtype)
            srccat = torch.cat([src, condition], dim=2)
            src = srccat.permute(1, 0, 2)
            transformer_out = self.net.transformer0(src, src_key_padding_mask=None)
            out0 = self.net.out0(transformer_out)
            out0 = out0.permute(1, 2, 0)
            return out0.permute(0, 2, 1)

        if raw_inputs.ndim != 3 or raw_inputs.shape[-1] != 8:
            raise ValueError(f"Expected emg (B,T,8), got {tuple(raw_inputs.shape)}")
        src = raw_inputs.permute(0, 2, 1)
        src = src.float() * math.sqrt(self.net.dim_model)
        src = torch.unsqueeze(src, dim=1).permute(0, 1, 2, 3)
        src = self.net.conv1(src)[:, :, 0, :].permute(0, 2, 1)
        z_cont = self.to_dcsa(src).unsqueeze(2)
        z_fused = self.dcsa(z_cont, x)
        src_fused = self.from_dcsa(z_fused[:, :, : z_cont.shape[2], :].mean(dim=2))
        src = self.net.positional_encoder(src_fused)
        condition = torch.ones(src.shape[0], src.shape[1], 2, device=src.device, dtype=src.dtype) * condval.reshape(
            condval.shape[0], 1, 1
        ).to(device=src.device, dtype=src.dtype)
        srccat = torch.cat([src, condition], dim=2)
        src = srccat.permute(1, 0, 2)
        transformer_out = self.net.transformer0(src, src_key_padding_mask=None)
        out0 = self.net.out0(transformer_out)
        out0 = out0.permute(1, 2, 0)
        return out0.permute(0, 2, 1)


def build_temporal_backbone(
    temporal_type: str,
    dim: int,
    *,
    task: Optional[str] = None,
    dst_cfg: Optional[DSTFormerConfig] = None,
    dst_v2: Optional[DSTFormerV2Config] = None,
    dst_v3_moe: Optional[DSTFormerV3MoEConfig] = None,
    dst_v4_dual_moe: Optional[DSTFormerV4DualMoEConfig] = None,
    dst_v5_guided_moe: Optional[DSTFormerV5GuidedMoEConfig] = None,
    tcn_cfg: Optional[TCNBackboneConfig] = None,
    dcsa_cfg: Optional[DCSAConfig] = None,
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
    if temporal_type in ("dstformer_v3_moe", "dst_v3_moe", "h6"):
        cfg_dict = {"dim": dim}
        if dst_cfg is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers"):
                if hasattr(dst_cfg, k): cfg_dict[k] = getattr(dst_cfg, k)
        if dst_v3_moe is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc", "num_experts"):
                if hasattr(dst_v3_moe, k): cfg_dict[k] = getattr(dst_v3_moe, k)
        cfg_dict.update({k: v for k, v in kwargs.items() if k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc", "num_experts")})
        cfg = DSTFormerV3MoEConfig(**cfg_dict)
        return DSTFormerV3MoETemporalBackbone(cfg)
    if temporal_type in ("dstformer_v4_dual_moe", "dst_v4_dual_moe", "h6_dual"):
        cfg_dict = {"dim": dim}
        if dst_cfg is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers"):
                if hasattr(dst_cfg, k): cfg_dict[k] = getattr(dst_cfg, k)
        if dst_v4_dual_moe is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc", "num_experts", "num_spatial_experts", "num_temporal_experts"):
                if hasattr(dst_v4_dual_moe, k): cfg_dict[k] = getattr(dst_v4_dual_moe, k)
        cfg_dict.update({k: v for k, v in kwargs.items() if k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc", "num_experts", "num_spatial_experts", "num_temporal_experts")})
        cfg = DSTFormerV4DualMoEConfig(**cfg_dict)
        return DSTFormerV4DualMoETemporalBackbone(cfg)
    if temporal_type in ("dstformer_v5_guided_moe", "dst_v5_guided_moe", "h8_guided"):
        cfg_dict = {"dim": dim}
        if dst_cfg is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers"):
                if hasattr(dst_cfg, k): cfg_dict[k] = getattr(dst_cfg, k)
        if dst_v5_guided_moe is not None:
            for k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc", "num_experts", "guide_mode"):
                if hasattr(dst_v5_guided_moe, k): cfg_dict[k] = getattr(dst_v5_guided_moe, k)
        cfg_dict.update({k: v for k, v in kwargs.items() if k in ("num_heads", "mlp_ratio", "dropout", "attn_dropout", "num_layers", "use_rope", "use_ltc", "num_experts", "guide_mode")})
        cfg = DSTFormerV5GuidedMoEConfig(**cfg_dict)
        return DSTFormerV5GuidedMoETemporalBackbone(cfg)
    if temporal_type in ("mia_official_transformer", "mia_official", "official_transformer", "mia_transformer"):
        if task is None:
            raise ValueError("temporal_type=official requires task to be provided")
        return MiaOfficialLightTransformerTemporal(task=task)
    if temporal_type in ("mia_official_transformer_conv_dcsa", "mia_official_conv_dcsa", "official_conv_dcsa"):
        if task is None:
            raise ValueError("temporal_type=official_conv_dcsa requires task to be provided")
        cfg = dcsa_cfg or DCSAConfig(dim=dim)
        return MiaOfficialLightTransformerConvDCSATemporal(task=task, dcsa_cfg=cfg)
    if temporal_type == "tcn":
        cfg = tcn_cfg or TCNBackboneConfig(dim=dim, **{k: v for k, v in kwargs.items() if k in ("hidden_dim", "kernel_size", "num_layers", "dilation_base", "dropout")})
        return TCNTemporalBackbone(cfg)
    raise ValueError(f"Unknown temporal_type={temporal_type!r}. Use 'dstformer' or 'tcn' or 'mia_official_transformer'.")
