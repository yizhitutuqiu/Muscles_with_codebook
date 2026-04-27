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


class SymmetricDCSA(nn.Module):
    """
    双向对称 DCSA（参考标准对齐流程）：
    1. 统一投影到共享隐空间
    2. 分别做双向交叉注意力 (Cont->Disc 和 Disc->Cont)
    3. 反向投影回各自原始维度
    4. 残差融合
    最后将两路特征在 Token 维度拼接 (Concat)，形成 (B, T, N_cont + N_disc, C) 送入后续处理。
    """
    def __init__(self, cfg: DCSAConfig):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.dim)
        
        # Step 1: 投影到共享隐空间 (尽管目前维度都是 dim，但加上投影层更符合理论推导并增加表达能力)
        self.proj_in_s = nn.Linear(d_model, d_model)
        self.proj_in_q = nn.Linear(d_model, d_model)
        
        # Step 2: 双向 Cross Attention
        self.attn_c2d = nn.MultiheadAttention(d_model, int(cfg.num_heads), float(cfg.dropout), batch_first=True)
        self.attn_d2c = nn.MultiheadAttention(d_model, int(cfg.num_heads), float(cfg.dropout), batch_first=True)
        
        # Step 3: 反向投影
        self.proj_out_s = nn.Linear(d_model, d_model)
        self.proj_out_q = nn.Linear(d_model, d_model)
        
        # Step 4: 残差融合使用的 Norm
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()

    def forward(self, cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
        """
        cont: (B, T, 25, C)
        disc: (B, T, 63, C)
        return: (B, T, 88, C) 拼接后的特征
        """
        b, t, j, c = cont.shape
        _, _, n, _ = disc.shape
        
        f_s = cont.reshape(b * t, j, c)
        f_q = disc.reshape(b * t, n, c)
        
        # Step 1: 投影
        f_s_proj = self.proj_in_s(f_s)
        f_q_proj = self.proj_in_q(f_q)
        
        # Step 2: 双向交叉注意力
        # C -> D (Query=连续骨骼特征, KV=离散Codebook)
        out_c2d, _ = self.attn_c2d(query=f_s_proj, key=f_q_proj, value=f_q_proj, need_weights=False)
        # D -> C (Query=离散Codebook, KV=连续骨骼特征)
        out_d2c, _ = self.attn_d2c(query=f_q_proj, key=f_s_proj, value=f_s_proj, need_weights=False)
        
        # Step 3: 反向投影
        out_s = self.proj_out_s(out_c2d)
        out_q = self.proj_out_q(out_d2c)
        
        # Step 4: 残差融合
        f_s_out = self.norm_s(f_s + self.drop(out_s)).reshape(b, t, j, c)
        f_q_out = self.norm_q(f_q + self.drop(out_q)).reshape(b, t, n, c)
        
        # 最后，将它们在 Token 维度拼接
        return torch.cat([f_s_out, f_q_out], dim=2)

