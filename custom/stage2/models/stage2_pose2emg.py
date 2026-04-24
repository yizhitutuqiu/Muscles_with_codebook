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


# 关节点数，与 joints3d (B,T,25,3) 一致
NUM_JOINTS: int = 25


@dataclass(frozen=True)
class Stage2Pose2EMGConfig:
    token_count: int = 63
    dim: int = 256
    # 连续分支：mixer=75 维→63 Token（原）；joint_25=25 关节各 3 维→25 Token（论文对齐，保留物理拓扑）
    cont_encoder_type: str = "mixer"
    cont_hidden_dim: int = 1024
    cont_joint_hidden_dim: int = 128  # joint_25 时小 MLP 的中间维，避免弱 Query 冷启动
    # 可插拔融合：dcsa（对称 63-63）| dcsa_asymmetric（Q=25,K/V=63）| residual_add
    fusion_type: str = "dcsa"
    dcsa: DCSAConfig = DCSAConfig()
    fusion_residual_add: Optional[ResidualAddConfig] = None
    # 可插拔融合后时序：dstformer | dstformer_v2 | dstformer_v3_moe | tcn
    temporal_type: str = "dstformer"
    dst: DSTFormerConfig = DSTFormerConfig()
    dst_v2: Optional[DSTFormerV2Config] = None
    dst_v3_moe: Optional[DSTFormerV3MoEConfig] = None
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
      joints3d sequence -> (1) discrete tokens via frozen stage1 codebook + (2) continuous tokens via trainable encoder
      -> 可插拔融合 (dcsa / residual_add) -> 可插拔时序 (dstformer / tcn) -> EMG regression.

    This module is self-contained and does NOT modify Stage I. Stage I model is injected and frozen.
    """

    def __init__(self, cfg: Stage2Pose2EMGConfig, *, stage1: FrameCodebookModel):
        super().__init__()
        self.cfg = cfg
        self.stage1 = stage1
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.stage1.eval()

        dim = int(cfg.dim)
        cont_type = str(cfg.cont_encoder_type).strip().lower()
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

        fused_token_count = self._cont_token_count if (
            cont_type in ("joint_25", "stgcn") and str(cfg.fusion_type).strip().lower() in ("dcsa_asymmetric", "dcsa_asym", "asymmetric_dcsa")
        ) else int(cfg.token_count)

        # H5: 离散 Token 同样需要时空位置编码，否则 DCSA 无法在 nocond 下进行稳定的空间对齐
        max_t = int(getattr(cfg, "max_seq_len", 256))
        self.disc_spatial_pe = nn.Parameter(torch.empty(1, 1, int(cfg.token_count), dim))
        self.disc_temporal_pe = nn.Parameter(torch.empty(1, max_t, 1, dim))
        nn.init.normal_(self.disc_spatial_pe, std=0.02)
        nn.init.normal_(self.disc_temporal_pe, std=0.02)

        self.fusion = build_fusion(
            cfg.fusion_type,
            dim,
            dcsa_cfg=cfg.dcsa,
            residual_add_cfg=cfg.fusion_residual_add,
        )
        self.temporal = build_temporal_backbone(
            cfg.temporal_type,
            dim,
            dst_cfg=cfg.dst,
            dst_v2=cfg.dst_v2,
            dst_v3_moe=cfg.dst_v3_moe,
            tcn_cfg=cfg.tcn,
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
        )
        self.emg_head = build_emg_head(cfg.emg_head_type, emg_cfg)
        if str(cfg.emg_pred_mode).strip().lower() == "residual":
            zero_init_final_output(self.emg_head)

    @torch.no_grad()
    def _stage1_discrete_tokens_bt(self, joints3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if joints3d.ndim != 4 or joints3d.shape[-2:] != (25, 3):
            raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
        b, t, _, _ = joints3d.shape
        mod = self.stage1.joints3d
        stage1_in_dim = int(mod.cfg.in_dim)
        stage1_token_count = int(mod.cfg.token_count)
        stage1_code_dim = int(mod.cfg.code_dim)
        if stage1_code_dim != int(self.cfg.dim):
            raise ValueError(f"Stage1 code_dim={stage1_code_dim} != Stage2 dim={int(self.cfg.dim)}")

        # Case A: unified clip-level codebook (e.g. clip_len=5, token_count=1, in_dim=clip_len*75)
        # We encode each clip into 1 token, then repeat it for all frames in that clip to form (B,T,token_count=1,dim).
        if stage1_token_count == 1 and stage1_in_dim % 75 == 0 and stage1_in_dim != 75:
            clip_len = stage1_in_dim // 75
            if int(t) % int(clip_len) != 0:
                raise ValueError(
                    f"Unified-clip Stage1 expects T divisible by clip_len. Got T={t}, clip_len={clip_len} "
                    f"(stage1.in_dim={stage1_in_dim}=clip_len*75)."
                )
            num_clips = int(t) // int(clip_len)
            x_clips = joints3d.reshape(b, num_clips, clip_len, 75).reshape(b * num_clips, stage1_in_dim)
            x_n = mod.normalize(x_clips.float(), update=False)
            z_e = mod.encoder(x_n).view(b * num_clips, 1, stage1_code_dim)
            z_q, idx, _, _, _ = self.stage1.vq(z_e.reshape(b * num_clips, stage1_code_dim))
            z_q_base = z_q.view(b * num_clips, stage1_token_count * stage1_code_dim)
            z_q = z_q.view(b, num_clips, 1, stage1_code_dim).repeat_interleave(int(clip_len), dim=1)
            idx_bt = idx.view(b, num_clips, 1).repeat_interleave(int(clip_len), dim=1)
            return z_q, idx_bt, z_q_base

        if stage1_in_dim == 75:
            if stage1_token_count != int(self.cfg.token_count):
                raise ValueError(
                    f"Stage1 token_count={stage1_token_count} != Stage2 token_count={int(self.cfg.token_count)}"
                )
            x_flat = joints3d.reshape(b * t, 75)
            x_n = mod.normalize(x_flat.float(), update=False)
            z_e = mod.encoder(x_n).view(b * t, stage1_token_count, stage1_code_dim)
            z_q, idx, _, _, _ = self.stage1.vq(z_e.reshape(b * t * stage1_token_count, stage1_code_dim))
            z_q = z_q.view(b, t, stage1_token_count, stage1_code_dim)
            idx_bt = idx.view(b, t, stage1_token_count)
            return z_q, idx_bt, None

        expected_clip_dim = int(t) * 75
        if stage1_in_dim != expected_clip_dim:
            raise ValueError(
                f"Unsupported Stage1 joints3d.in_dim={stage1_in_dim}. "
                f"Expected 75 (per-frame), {expected_clip_dim} (frame-aligned clip={t}*75), "
                f"or clip_len*75 with token_count=1 (unified clip-level codebook)."
            )
        if stage1_token_count % int(t) != 0:
            raise ValueError(f"Stage1 token_count={stage1_token_count} not divisible by T={t}")
        tokens_per_frame = stage1_token_count // int(t)
        if tokens_per_frame != int(self.cfg.token_count):
            raise ValueError(
                f"Stage1 tokens_per_frame={tokens_per_frame} != Stage2 token_count={int(self.cfg.token_count)}. "
                "For frame-aligned temporal codebook, Stage2 token_count must equal Stage1 token_count/T."
            )

        x_clip = joints3d.reshape(b, expected_clip_dim)
        x_n = mod.normalize(x_clip.float(), update=False)
        z_e = mod.encoder(x_n).view(b, stage1_token_count, stage1_code_dim)
        z_q, idx, _, _, _ = self.stage1.vq(z_e.reshape(b * stage1_token_count, stage1_code_dim))
        z_q = z_q.view(b, stage1_token_count, stage1_code_dim)
        idx = idx.view(b, stage1_token_count)
        z_q_bt = z_q.view(b, t, tokens_per_frame, stage1_code_dim)
        idx_bt = idx.view(b, t, tokens_per_frame)
        z_q_clip_flat = z_q.reshape(b, stage1_token_count * stage1_code_dim)
        return z_q_bt, idx_bt, z_q_clip_flat

    def forward(self, joints3d: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        时序网络：T 帧一起输入；每帧分别走离散/连续两路 + DCSA 得到融合特征，再送入 DSTFormer 做时空提取。

        joints3d: (B, T, 25, 3)  (e.g. T=30; assumed already root-centered if desired)
        cond: (B, 1) or (B,) or None
        return dict with:
          emg_pred: (B, T, 8)
          idx_j3d:  (B, T, token_count)  code indices for analysis
        """
        if joints3d.ndim != 4 or joints3d.shape[-2:] != (25, 3):
            raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
        b, t, _, _ = joints3d.shape
        x_flat = joints3d.reshape(b * t, 75)

        z_disc_bt, idx_bt, z_disc_clip_flat = self._stage1_discrete_tokens_bt(joints3d)

        # Branch B: continuous tokens
        n_cont = self._cont_token_count
        cont_type = str(self.cfg.cont_encoder_type).strip().lower()
        if cont_type in ("stgcn", "joint_25"):
            x_joints = x_flat.view(b * t, NUM_JOINTS, 3)
            z_cont = self.cont_encoder(x_joints)
        else:
            z_cont = self.cont_encoder(x_flat).view(b * t, n_cont, int(self.cfg.dim))

        z_cont_bt = z_cont.view(b, t, n_cont, int(self.cfg.dim))
        # 显式注入位置编码，让连续 Token 拥有空间身份和时间顺序
        if self.spatial_pe is not None and self.temporal_pe is not None:
            z_cont_bt = z_cont_bt + self.spatial_pe + self.temporal_pe[:, :t, :, :]
            
        # H5: 为离散 Token 注入时空位置编码
        z_disc_bt = z_disc_bt + self.disc_spatial_pe + self.disc_temporal_pe[:, :t, :, :]

        z_fused = self.fusion(z_cont_bt, z_disc_bt)

        if self.cond_proj is not None and cond is not None:
            cond = cond.view(b, 1)
            cond_feat = self.cond_proj(cond).view(b, 1, 1, int(self.cfg.dim))
            z_fused = z_fused + cond_feat

        # 融合后时序，可插拔：dstformer / tcn -> (B,T,N,C)，保留 N 不做 mean
        z_out = self.temporal(z_fused)
        emg_out = self.emg_head(z_out)  # (B,T,8)，EMG 头输出：全量或残差由 emg_pred_mode 决定

        pred_mode = str(self.cfg.emg_pred_mode).strip().lower()
        if pred_mode == "residual":
            if self.stage1.emg is None:
                raise RuntimeError("emg_pred_mode=residual 需要 Stage1 含有 EMG 模态（j3d->emg 交叉 decoder），当前 Stage1 无 emg。")
            # Stage1 交叉重建 j3d->z_disc->emg.decoder 作为保底（归一化空间），反归一化到与 gt 一致
            with torch.no_grad():
                stage1_emg_in_dim = int(self.stage1.emg.cfg.in_dim)
                if stage1_emg_in_dim == 8:
                    n, c = int(self.cfg.token_count), int(self.cfg.dim)
                    z_flat = z_disc_bt.reshape(b * t, n * c)
                    emg_base = self.stage1.emg.decoder(z_flat)
                    if self.stage1.emg.standardizer is not None:
                        std = torch.sqrt(self.stage1.emg.standardizer.var + 1e-6)
                        emg_base = emg_base * std + self.stage1.emg.standardizer.mean
                    emg_base = emg_base.view(b, t, 8)
                else:
                    if z_disc_clip_flat is None:
                        raise RuntimeError("Stage1 emg is clip-wise but clip tokens are missing.")
                    emg_base = self.stage1.emg.decoder(z_disc_clip_flat)
                    if self.stage1.emg.standardizer is not None:
                        std = torch.sqrt(self.stage1.emg.standardizer.var + 1e-6)
                        emg_base = emg_base * std + self.stage1.emg.standardizer.mean
                    emg_base = emg_base.view(b, t, 8)
            emg_pred = emg_base + emg_out
        else:
            emg_pred = emg_out

        out = {
            "emg_pred": emg_pred,
            "idx_j3d": idx_bt,
        }
        return out
