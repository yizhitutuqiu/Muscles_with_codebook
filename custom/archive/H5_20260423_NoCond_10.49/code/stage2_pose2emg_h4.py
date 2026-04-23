import torch
import torch.nn as nn
from typing import Dict, Optional

from .stage2_pose2emg import Stage2Pose2EMG, Stage2Pose2EMGConfig
from custom.models.frame_codebook import FrameCodebookModel
from custom.models.mlp import MLP

class Stage2Pose2EMG_H4(Stage2Pose2EMG):
    """
    H4 实验重构版: 
    由于在 no_cond 的设定下，网络失去了明确的 Subject Identity 偏置，
    如果只是简单地去掉 cond_proj，整个高层语义表示空间将变得扁平。
    
    H4 假设：
    引入可学习的隐式全局语义 Token (Implicit Semantic Query)。
    在缺乏 Cond 的情况下，让模型自主学习出一组基于全局动作规律的辅助 Query 参与 DCSA/DSTFormer 计算。
    """
    def __init__(self, cfg: Stage2Pose2EMGConfig, *, stage1: FrameCodebookModel):
        super().__init__(cfg, stage1=stage1)
        
        # 如果模型不使用 Cond (no_cond)，我们强行引入一个可学习的隐式偏置 (Implicit Bias)
        if not cfg.use_cond:
            # 一个全局的隐式特征，等同于无 cond 时的默认 'identity' 特征
            self.implicit_bias = nn.Parameter(torch.zeros(1, 1, 1, cfg.dim))
            nn.init.normal_(self.implicit_bias, std=0.02)
            
            # 增加一个非线性投影层来强化这种隐式特征的表达
            self.implicit_proj = nn.Sequential(
                nn.Linear(cfg.dim, cfg.dim * 2),
                nn.GELU(),
                nn.Linear(cfg.dim * 2, cfg.dim)
            )
            
    def forward(self, joints3d: torch.Tensor, cond: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if joints3d.ndim != 4 or joints3d.shape[-2:] != (25, 3):
            raise ValueError(f"Expected joints3d (B,T,25,3), got {tuple(joints3d.shape)}")
        
        b, t, _, _ = joints3d.shape
        x_flat = joints3d.reshape(b * t, 75)

        # 1. 离散分支
        z_disc_bt, idx_bt, _ = self._stage1_discrete_tokens_bt(joints3d)

        # 2. 连续分支
        n_cont = self._cont_token_count
        if n_cont == 25:
            x_joints = x_flat.view(b * t, 25, 3)
            z_cont = self.cont_encoder(x_joints)
        else:
            z_cont = self.cont_encoder(x_flat).view(b * t, n_cont, int(self.cfg.dim))

        z_cont_bt = z_cont.view(b, t, n_cont, int(self.cfg.dim))
        
        if self.spatial_pe is not None and self.temporal_pe is not None:
            z_cont_bt = z_cont_bt + self.spatial_pe + self.temporal_pe[:, :t, :, :]
            
        # 3. 融合层
        z_fused = self.fusion(z_cont_bt, z_disc_bt)

        # 4. H4 核心改动：Implicit Bias 注入
        if self.cond_proj is not None and cond is not None:
            cond_feat = self.cond_proj(cond.view(b, 1)).view(b, 1, 1, int(self.cfg.dim))
            z_fused = z_fused + cond_feat
        elif hasattr(self, 'implicit_bias'):
            # 如果是 no_cond 模式，使用学习到的 implicit_bias
            imp_feat = self.implicit_proj(self.implicit_bias)
            z_fused = z_fused + imp_feat

        # 5. 时序主干与回归头
        z_out = self.temporal(z_fused)
        emg_out = self.emg_head(z_out)

        pred_mode = str(self.cfg.emg_pred_mode).strip().lower()
        if pred_mode == "full":
            return {"emg_pred": emg_out, "idx_j3d": idx_bt}
        else:
            raise NotImplementedError("H4 refactor focuses on 'full' mode.")
