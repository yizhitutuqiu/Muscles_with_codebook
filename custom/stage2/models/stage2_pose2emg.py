from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...models.frame_codebook import FrameCodebookModel
from ...models.mlp import MLPMixerConfig, FourLayerMLPMixer
from .dcsa import DCSAConfig
from .emg_head import EMGHeadConfig, build_emg_head, zero_init_final_output
from .fusion import ResidualAddConfig, build_fusion
from .temporal_backbone import TCNBackboneConfig, build_temporal_backbone
from .dstformer import DSTFormerConfig
from .dstformer_v2 import DSTFormerV2Config
from .dstformer_v3_moe import DSTFormerV3MoEConfig
from .dstformer_v4_dual_moe import DSTFormerV4DualMoEConfig
from .dstformer_v5_guided_moe import DSTFormerV5GuidedMoEConfig


# 关节点数，与 joints3d (B,T,25,3) 一致
NUM_JOINTS: int = 25


@dataclass(frozen=True)
class Stage2Pose2EMGConfig:
    task: str = "pose2emg"  # "pose2emg" or "emg2pose"
    token_count: int = 63
    dim: int = 256
    disc_fusion: str = "single"  # single | concat | hierarchical
    # 连续分支：mixer=75 维→63 Token（原）；joint_25=25 关节各 3 维→25 Token（论文对齐，保留物理拓扑）
    cont_encoder_type: str = "mixer"
    cont_hidden_dim: int = 1024
    cont_joint_hidden_dim: int = 128  # joint_25 时小 MLP 的中间维，避免弱 Query 冷启动
    # 可插拔融合：dcsa（对称 63-63）| dcsa_asymmetric（Q=25,K/V=63）| residual_add
    fusion_type: str = "dcsa"
    dcsa: DCSAConfig = DCSAConfig()
    fusion_residual_add: Optional[ResidualAddConfig] = None
    # 可插拔融合后时序：dstformer | dstformer_v2 | dstformer_v3_moe | dstformer_v4_dual_moe | tcn
    temporal_type: str = "dstformer"
    dst: DSTFormerConfig = DSTFormerConfig()
    dst_v2: Optional[DSTFormerV2Config] = None
    dst_v3_moe: Optional[DSTFormerV3MoEConfig] = None
    dst_v4_dual_moe: Optional[DSTFormerV4DualMoEConfig] = None
    dst_v5_guided_moe: Optional[DSTFormerV5GuidedMoEConfig] = None
    tcn: Optional[TCNBackboneConfig] = None
    # EMG 头：对 N 个 token 做融合后映射到 8 维。mixer=Stage1 式 Mixer（推荐）/ flatten=展平+Linear
    emg_head_type: str = "mixer"
    emg_head: Optional[EMGHeadConfig] = None
    emg_hidden: int = 256
    emg_mixer_hidden_dim: int = 256
    emg_mixer_num_layers: int = 4
    # 输出策略：full=Stage2 直接预测 8 维；residual=Stage2 预测修正值，最终 emg_pred = Stage1 交叉重建 + 修正
    emg_pred_mode: str = "full"
    # 时空位置编码：joint_25 时在 z_cont 送入 DCSA 前注入，时间维最大长度
    max_seq_len: int = 256
    use_cond: bool = False


class Stage2Pose2EMG(nn.Module):
    """
    Stage II:
      (pose2emg) joints3d sequence -> discrete tokens + continuous tokens -> fusion -> temporal -> EMG regression.
      (emg2pose) emg sequence -> discrete tokens + continuous tokens -> fusion -> temporal -> joints3d regression.

    This module is self-contained and does NOT modify Stage I. Stage I model is injected and frozen.
    """

    def __init__(
        self,
        cfg: Stage2Pose2EMGConfig,
        *,
        stage1: FrameCodebookModel,
        stage1_aux: Optional[FrameCodebookModel] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.task = str(getattr(cfg, "task", "pose2emg")).strip().lower()
        self.stage1 = stage1
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.stage1.eval()
        self.stage1_aux = stage1_aux
        if self.stage1_aux is not None:
            for p in self.stage1_aux.parameters():
                p.requires_grad_(False)
            self.stage1_aux.eval()

        dim = int(cfg.dim)
        cont_type = str(cfg.cont_encoder_type).strip().lower()
        
        if self.task == "emg2pose":
            # For EMG to Pose, we treat EMG 8 channels as input
            # If they requested 'joint_25' or 'emg_8', we use an 8-token spatial layout
            if cont_type in ("emg_8", "joint_25"):
                joint_hidden = int(cfg.cont_joint_hidden_dim)
                self.cont_encoder = nn.Sequential(
                    nn.Linear(1, joint_hidden),
                    nn.GELU(),
                    nn.Linear(joint_hidden, dim),
                    nn.LayerNorm(dim),
                )
                self._cont_token_count = 8
                max_t = int(getattr(cfg, "max_seq_len", 256))
                self.spatial_pe = nn.Parameter(torch.empty(1, 1, 8, dim))
                self.temporal_pe = nn.Parameter(torch.empty(1, max_t, 1, dim))
                nn.init.normal_(self.spatial_pe, std=0.02)
                nn.init.normal_(self.temporal_pe, std=0.02)
            else:
                # Default to MLP Mixer treating the 8 channels as flat input
                self.spatial_pe = None
                self.temporal_pe = None
                mixer_cfg = MLPMixerConfig(
                    in_dim=8,
                    token_count=int(cfg.token_count),
                    code_dim=dim,
                    hidden_dim=int(cfg.cont_hidden_dim),
                    num_layers=4,
                )
                self.cont_encoder = FourLayerMLPMixer(mixer_cfg)
                self._cont_token_count = int(cfg.token_count)
        else:
            if cont_type == "joint_25":
                # 小 MLP + LayerNorm，避免“弱 Query”冷启动：单层 Linear 无法提出有意义的 Query
                joint_hidden = int(cfg.cont_joint_hidden_dim)
                self.cont_encoder = nn.Sequential(
                    nn.Linear(3, joint_hidden),
                    nn.GELU(),
                    nn.Linear(joint_hidden, dim),
                    nn.LayerNorm(dim),
                )
                self._cont_token_count = NUM_JOINTS
                # 时空位置编码：让 25 个 Token 拥有空间身份和时间顺序，在送入 DCSA 前注入
                max_t = int(getattr(cfg, "max_seq_len", 256))
                self.spatial_pe = nn.Parameter(torch.empty(1, 1, NUM_JOINTS, dim))
                self.temporal_pe = nn.Parameter(torch.empty(1, max_t, 1, dim))
                nn.init.normal_(self.spatial_pe, std=0.02)
                nn.init.normal_(self.temporal_pe, std=0.02)
            elif cont_type == "stgcn":
                # H6-B: ST-GCN continuous encoder
                from .st_gcn import AdaptiveGCNEncoder
                self.cont_encoder = AdaptiveGCNEncoder(
                    in_channels=3, 
                    hidden_dim=int(cfg.cont_joint_hidden_dim), 
                    out_dim=dim, 
                    num_nodes=NUM_JOINTS, 
                    num_layers=3
                )
                self._cont_token_count = NUM_JOINTS
                # 时空位置编码
                max_t = int(getattr(cfg, "max_seq_len", 256))
                self.spatial_pe = nn.Parameter(torch.empty(1, 1, NUM_JOINTS, dim))
                self.temporal_pe = nn.Parameter(torch.empty(1, max_t, 1, dim))
                nn.init.normal_(self.spatial_pe, std=0.02)
                nn.init.normal_(self.temporal_pe, std=0.02)
            else:
                self.spatial_pe = None
                self.temporal_pe = None
                mixer_cfg = MLPMixerConfig(
                    in_dim=75,
                    token_count=int(cfg.token_count),
                    code_dim=dim,
                    hidden_dim=int(cfg.cont_hidden_dim),
                    num_layers=4,
                )
                self.cont_encoder = FourLayerMLPMixer(mixer_cfg)
                self._cont_token_count = int(cfg.token_count)

        fusion_type_str = str(cfg.fusion_type).strip().lower()
        if fusion_type_str in ("dcsa_asymmetric", "dcsa_asym", "asymmetric_dcsa"):
            fused_token_count = self._cont_token_count
        elif fusion_type_str in ("dcsa_symmetric", "dcsa_sym", "symmetric_dcsa"):
            fused_token_count = self._cont_token_count + int(cfg.token_count)
        elif fusion_type_str in ("none", "continuous_only"):
            fused_token_count = self._cont_token_count
        else:
            fused_token_count = int(cfg.token_count)

        # H5: 离散 Token 同样需要时空位置编码，否则 DCSA 无法在 nocond 下进行稳定的空间对齐
        max_t = int(getattr(cfg, "max_seq_len", 256))
        self.disc_spatial_pe = nn.Parameter(torch.empty(1, 1, int(cfg.token_count), dim))
        self.disc_temporal_pe = nn.Parameter(torch.empty(1, max_t, 1, dim))
        nn.init.normal_(self.disc_spatial_pe, std=0.02)
        nn.init.normal_(self.disc_temporal_pe, std=0.02)

        disc_fusion = str(getattr(cfg, "disc_fusion", "single")).strip().lower()
        self.disc_fusion = disc_fusion
        if disc_fusion == "hierarchical":
            self.fusion1 = build_fusion(
                cfg.fusion_type,
                dim,
                dcsa_cfg=cfg.dcsa,
                residual_add_cfg=cfg.fusion_residual_add,
            )
            self.fusion2 = build_fusion(
                cfg.fusion_type,
                dim,
                dcsa_cfg=cfg.dcsa,
                residual_add_cfg=cfg.fusion_residual_add,
            )
            self.fusion = None
        else:
            self.fusion = build_fusion(
                cfg.fusion_type,
                dim,
                dcsa_cfg=cfg.dcsa,
                residual_add_cfg=cfg.fusion_residual_add,
            )
            self.fusion1 = None
            self.fusion2 = None
        self.temporal = build_temporal_backbone(
            cfg.temporal_type,
            dim=dim,
            task=self.task,
            dst_cfg=cfg.dst,
            dst_v2=cfg.dst_v2,
            dst_v3_moe=cfg.dst_v3_moe,
            dst_v4_dual_moe=cfg.dst_v4_dual_moe,
            dst_v5_guided_moe=cfg.dst_v5_guided_moe,
            tcn_cfg=cfg.tcn,
            dcsa_cfg=cfg.dcsa,
        )

        if getattr(cfg, "use_cond", False):
            self.cond_proj = nn.Sequential(
                nn.Linear(1, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
        else:
            self.cond_proj = None

        emg_cfg = cfg.emg_head or EMGHeadConfig(
            token_count=fused_token_count,
            dim=dim,
            mixer_hidden_dim=int(cfg.emg_mixer_hidden_dim),
            mixer_num_layers=int(cfg.emg_mixer_num_layers),
            flatten_hidden_dim=int(cfg.emg_hidden),
            out_dim=8 if self.task == "pose2emg" else 75
        )
        self.emg_head = build_emg_head(cfg.emg_head_type, emg_cfg)
        if str(cfg.emg_pred_mode).strip().lower() == "residual":
            zero_init_final_output(self.emg_head)

    @torch.no_grad()
    def _stage1_discrete_tokens_bt_for(
        self, stage1: FrameCodebookModel, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # inputs can be joints3d (B,T,25,3) or emg (B,T,8) depending on task
        b, t = inputs.shape[0], inputs.shape[1]
        
        if self.task == "pose2emg":
            if inputs.ndim != 4 or inputs.shape[-2:] != (25, 3):
                raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(inputs.shape)}")
            mod = stage1.joints3d
            expected_frame_dim = 75
            x_flat = inputs.reshape(b * t, 75)
        else:
            if inputs.ndim != 3 or inputs.shape[-1] != 8:
                raise ValueError(f"Expected emg (B,T,8), got {tuple(inputs.shape)}")
            mod = stage1.emg
            expected_frame_dim = 8
            x_flat = inputs.reshape(b * t, 8)

        stage1_in_dim = int(mod.cfg.in_dim)
        stage1_token_count = int(mod.cfg.token_count)
        stage1_code_dim = int(mod.cfg.code_dim)
        if stage1_code_dim != int(self.cfg.dim):
            raise ValueError(f"Stage1 code_dim={stage1_code_dim} != Stage2 dim={int(self.cfg.dim)}")
        
        bypass_vq = bool(getattr(getattr(stage1, "cfg", None), "encoder_decoder_only", False))

        # Case A: unified clip-level codebook
        if stage1_token_count == 1 and stage1_in_dim % expected_frame_dim == 0 and stage1_in_dim != expected_frame_dim:
            clip_len = stage1_in_dim // expected_frame_dim
            if int(t) % int(clip_len) != 0:
                raise ValueError(
                    f"Unified-clip Stage1 expects T divisible by clip_len. Got T={t}, clip_len={clip_len}"
                )
            num_clips = int(t) // int(clip_len)
            x_clips = inputs.reshape(b, num_clips, clip_len, expected_frame_dim).reshape(b * num_clips, stage1_in_dim)
            x_n = mod.normalize(x_clips.float(), update=False)
            z_e = mod.encoder(x_n).view(b * num_clips, 1, stage1_code_dim)
            if bypass_vq:
                dev = z_e.device
                idx = torch.zeros((b * num_clips,), dtype=torch.long, device=dev)
                z_q_base = z_e.view(b * num_clips, stage1_code_dim)
                z_q = z_e.view(b, num_clips, 1, stage1_code_dim).repeat_interleave(int(clip_len), dim=1)
                idx_bt = idx.view(b, num_clips, 1).repeat_interleave(int(clip_len), dim=1)
                return z_q, idx_bt, z_q_base
            else:
                z_q, idx, _, _, _ = stage1.vq(z_e.reshape(b * num_clips, stage1_code_dim))
                z_q_base = z_q.view(b * num_clips, stage1_token_count * stage1_code_dim)
                z_q = z_q.view(b, num_clips, 1, stage1_code_dim).repeat_interleave(int(clip_len), dim=1)
                idx_bt = idx.view(b, num_clips, 1).repeat_interleave(int(clip_len), dim=1)
                return z_q, idx_bt, z_q_base

        if stage1_in_dim == expected_frame_dim:
            x_n = mod.normalize(x_flat.float(), update=False)
            z_e = mod.encoder(x_n).view(b * t, stage1_token_count, stage1_code_dim)
            if bypass_vq:
                dev = z_e.device
                z_q = z_e.view(b, t, stage1_token_count, stage1_code_dim)
                idx_bt = torch.zeros((b, t, stage1_token_count), dtype=torch.long, device=dev)
                return z_q, idx_bt, None
            else:
                z_q, idx, _, _, _ = stage1.vq(z_e.reshape(b * t * stage1_token_count, stage1_code_dim))
                z_q = z_q.view(b, t, stage1_token_count, stage1_code_dim)
                idx_bt = idx.view(b, t, stage1_token_count)
                return z_q, idx_bt, None

        expected_clip_dim = int(t) * expected_frame_dim
        if stage1_in_dim != expected_clip_dim:
            raise ValueError(
                f"Unsupported Stage1 in_dim={stage1_in_dim}. "
                f"Expected {expected_frame_dim} (per-frame), {expected_clip_dim} (frame-aligned clip)"
            )
            
        x_clip = inputs.reshape(b, expected_clip_dim)
        x_n = mod.normalize(x_clip.float(), update=False)
        z_e = mod.encoder(x_n).view(b, stage1_token_count, stage1_code_dim)
        if bypass_vq:
            dev = z_e.device
            z_q = z_e
            idx = torch.zeros((b, stage1_token_count), dtype=torch.long, device=dev)
        else:
            z_q, idx, _, _, _ = stage1.vq(z_e.reshape(b * stage1_token_count, stage1_code_dim))
            z_q = z_q.view(b, stage1_token_count, stage1_code_dim)
            idx = idx.view(b, stage1_token_count)
        
        if stage1_token_count % int(t) != 0 and stage1_token_count != 1:
            tokens_per_frame = stage1_token_count
            z_q_bt = z_q.unsqueeze(1).expand(b, t, tokens_per_frame, stage1_code_dim)
            idx_bt = idx.unsqueeze(1).expand(b, t, tokens_per_frame)
        else:
            tokens_per_frame = stage1_token_count // int(t)
            if tokens_per_frame != int(self.cfg.token_count):
                raise ValueError(
                    f"Stage1 tokens_per_frame={tokens_per_frame} != Stage2 token_count={int(self.cfg.token_count)}"
                )
            z_q_bt = z_q.view(b, t, tokens_per_frame, stage1_code_dim)
            idx_bt = idx.view(b, t, tokens_per_frame)

        z_q_clip_flat = z_q.reshape(b, stage1_token_count * stage1_code_dim)
        return z_q_bt, idx_bt, z_q_clip_flat

    @torch.no_grad()
    def _stage1_discrete_tokens_bt(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.stage1_aux is None:
            return self._stage1_discrete_tokens_bt_for(self.stage1, inputs)

        disc_fusion = str(getattr(self.cfg, "disc_fusion", "single")).strip().lower()
        if disc_fusion != "concat":
            return self._stage1_discrete_tokens_bt_for(self.stage1, inputs)

        z1, idx1, _ = self._stage1_discrete_tokens_bt_for(self.stage1_aux, inputs)
        z5, idx5, _ = self._stage1_discrete_tokens_bt_for(self.stage1, inputs)
        z = torch.cat([z1, z5], dim=2)
        idx = torch.cat([idx1, idx5], dim=2)
        return z, idx, None

    def forward(self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        时序网络：T 帧一起输入；每帧分别走离散/连续两路 + DCSA 得到融合特征，再送入 DSTFormer 做时空提取。

        inputs: (B, T, 25, 3) for pose2emg or (B, T, 8) for emg2pose
        cond: (B, 1) or (B,) or None
        return dict with:
          pred: (B, T, 8) for pose2emg or (B, T, 25, 3) for emg2pose
          idx_j3d:  (B, T, token_count)  code indices for analysis
        """
        b, t = inputs.shape[0], inputs.shape[1]
        
        if self.task == "pose2emg":
            if inputs.ndim != 4 or inputs.shape[-2:] != (25, 3):
                raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(inputs.shape)}")
            x_flat = inputs.reshape(b * t, 75)
        else:
            if inputs.ndim != 3 or inputs.shape[-1] != 8:
                raise ValueError(f"Expected emg (B,T,8), got {tuple(inputs.shape)}")
            x_flat = inputs.reshape(b * t, 8)

        if self.stage1_aux is not None and str(getattr(self.cfg, "disc_fusion", "single")).strip().lower() == "hierarchical":
            z_disc1_bt, idx1_bt, _ = self._stage1_discrete_tokens_bt_for(self.stage1_aux, inputs)
            z_disc5_bt, idx5_bt, _ = self._stage1_discrete_tokens_bt_for(self.stage1, inputs)
            idx_bt = torch.cat([idx1_bt, idx5_bt], dim=2)
            z_disc_bt = None
        else:
            z_disc_bt, idx_bt, _ = self._stage1_discrete_tokens_bt(inputs)

        if bool(getattr(self.temporal, "expects_disc_tokens", False)):
            if z_disc_bt is None:
                raise ValueError("temporal expects discrete tokens but got z_disc_bt=None")
            z_disc_bt = z_disc_bt + self.disc_spatial_pe + self.disc_temporal_pe[:, :t, :, :]
            import inspect
            sig = inspect.signature(self.temporal.forward)
            kwargs = {}
            if "raw_inputs" in sig.parameters:
                kwargs["raw_inputs"] = inputs
            if "cond" in sig.parameters:
                kwargs["cond"] = cond
            z_out = self.temporal(z_disc_bt, **kwargs)
            if bool(getattr(self.temporal, "produces_pred", False)):
                net_out = z_out
            else:
                net_out = self.emg_head(z_out)
            pred_mode = str(self.cfg.emg_pred_mode).strip().lower()
            if pred_mode == "residual":
                raise NotImplementedError("residual mode is not fully adapted for emg2pose yet.")
            else:
                pred = net_out
            if self.task == "emg2pose":
                pred = pred.view(b, t, 25, 3)
            return {
                "pred": pred,
                "idx_j3d": idx_bt,
            }

        # Branch B: continuous tokens
        n_cont = self._cont_token_count
        cont_type = str(self.cfg.cont_encoder_type).strip().lower()
        
        if self.task == "pose2emg" and cont_type in ("stgcn", "joint_25"):
            x_joints = x_flat.view(b * t, NUM_JOINTS, 3)
            z_cont = self.cont_encoder(x_joints)
        elif self.task == "emg2pose" and cont_type in ("emg_8", "joint_25"):
            x_emg = x_flat.view(b * t, 8, 1)
            z_cont = self.cont_encoder(x_emg)
        else:
            z_cont = self.cont_encoder(x_flat).view(b * t, n_cont, int(self.cfg.dim))

        z_cont_bt = z_cont.view(b, t, n_cont, int(self.cfg.dim))
        # 显式注入位置编码
        if self.spatial_pe is not None and self.temporal_pe is not None:
            z_cont_bt = z_cont_bt + self.spatial_pe + self.temporal_pe[:, :t, :, :]
            
        if self.stage1_aux is not None and str(getattr(self.cfg, "disc_fusion", "single")).strip().lower() == "hierarchical":
            z_disc1_bt = z_disc1_bt + self.disc_spatial_pe[:, :, 0:1, :] + self.disc_temporal_pe[:, :t, :, :]
            z_disc5_bt = z_disc5_bt + self.disc_spatial_pe[:, :, 1:2, :] + self.disc_temporal_pe[:, :t, :, :]
            z_fused1 = self.fusion1(z_cont_bt, z_disc1_bt)
            z_fused = self.fusion2(z_fused1, z_disc5_bt)
            guide_feat = torch.cat([z_disc1_bt, z_disc5_bt], dim=2).mean(dim=2)
        else:
            z_disc_bt = z_disc_bt + self.disc_spatial_pe + self.disc_temporal_pe[:, :t, :, :]
            z_fused = self.fusion(z_cont_bt, z_disc_bt)
            guide_feat = z_disc_bt.mean(dim=2) if z_disc_bt is not None else None

        if self.cond_proj is not None and cond is not None:
            cond = cond.view(b, 1)
            cond_feat = self.cond_proj(cond).view(b, 1, 1, int(self.cfg.dim))
            z_fused = z_fused + cond_feat

        # 融合后时序，可插拔：dstformer / tcn -> (B,T,N,C)
        if hasattr(self.temporal, "forward"):
            import inspect
            sig = inspect.signature(self.temporal.forward)
            kwargs = {}
            if "guide" in sig.parameters:
                kwargs["guide"] = guide_feat
            if "raw_inputs" in sig.parameters:
                kwargs["raw_inputs"] = inputs
            if "cond" in sig.parameters:
                kwargs["cond"] = cond
            z_out = self.temporal(z_fused, **kwargs)
        else:
            z_out = self.temporal(z_fused)
            
        if bool(getattr(self.temporal, "produces_pred", False)):
            net_out = z_out
        else:
            net_out = self.emg_head(z_out)  # (B,T,8) or (B,T,75)

        pred_mode = str(self.cfg.emg_pred_mode).strip().lower()
        if pred_mode == "residual":
            raise NotImplementedError("residual mode is not fully adapted for emg2pose yet.")
        else:
            pred = net_out
            
        if self.task == "emg2pose":
            pred = pred.view(b, t, 25, 3)

        out = {
            "pred": pred,
            "idx_j3d": idx_bt,
        }
        return out
