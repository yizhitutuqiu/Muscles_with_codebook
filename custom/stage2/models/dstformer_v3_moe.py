from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dstformer_v2 import apply_rotary_emb, precompute_freqs_cis, SwiGLU

@dataclass(frozen=True)
class DSTFormerV3MoEConfig:
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    num_layers: int = 4
    use_rope: bool = True
    use_ltc: bool = True  # Local Temporal Convolution
    num_experts: int = 4  # Kinematic-driven MoE routing

class DenseMoESwiGLU(nn.Module):
    """
    Soft Mixture of Experts (Dense MoE) over SwiGLU.
    Instead of hard routing, we compute all experts and softly mix them based on router probabilities.
    This maintains gradient flow easily and natively supports 1D convolutions (LTC) since the sequence is unbroken.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, 
                 num_experts: int = 4, dropout: float = 0.0, use_ltc: bool = False):
        super().__init__()
        self.num_experts = num_experts
        # Router predicts expert weights for each token
        self.router = nn.Linear(in_features, num_experts)
        
        self.experts = nn.ModuleList([
            SwiGLU(in_features, hidden_features, out_features, dropout, use_ltc)
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, is_temporal: bool = False) -> torch.Tensor:
        # x: (B, L, C)
        router_logits = self.router(x)  # (B, L, E)
        routing_weights = F.softmax(router_logits, dim=-1)  # (B, L, E)
        
        out = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x, is_temporal=is_temporal)  # (B, L, C)
            # Mix expert output with its corresponding weight
            out = out + expert_out * routing_weights[..., i:i+1]
            
        return out

class _DSTFormerV3MoEBlock(nn.Module):
    def __init__(self, cfg: DSTFormerV3MoEConfig, max_t: int = 1024):
        super().__init__()
        dim = int(cfg.dim)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = dim // self.num_heads
        
        # Spatial
        self.s_norm = nn.LayerNorm(dim)
        self.s_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=float(cfg.attn_dropout),
            batch_first=True,
        )
        self.s_ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * float(cfg.mlp_ratio))
        # 2/3 hidden for SwiGLU
        swiglu_hidden = int(hidden * 2 / 3)
        # H6-C: Spatial FFN uses Dense MoE
        self.s_ffn = DenseMoESwiGLU(dim, swiglu_hidden, dim, cfg.num_experts, cfg.dropout, use_ltc=False)
        
        # Temporal
        self.t_norm = nn.LayerNorm(dim)
        self.t_qkv = nn.Linear(dim, dim * 3)
        self.t_proj = nn.Linear(dim, dim)
        self.t_attn_drop = nn.Dropout(float(cfg.attn_dropout))
        
        self.t_ffn_norm = nn.LayerNorm(dim)
        # H6-C: Temporal FFN uses Dense MoE (with LTC support)
        self.t_ffn = DenseMoESwiGLU(dim, swiglu_hidden, dim, cfg.num_experts, cfg.dropout, use_ltc=cfg.use_ltc)
        
        self.drop = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()
        self.use_rope = cfg.use_rope
        
        if self.use_rope:
            freqs_cis = precompute_freqs_cis(self.head_dim, max_t)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,C)
        b, t, n, c = x.shape
        
        # --- Spatial Attention ---
        xs = self.s_norm(x).reshape(b * t, n, c)
        ys, _ = self.s_attn(query=xs, key=xs, value=xs, need_weights=False)
        ys = ys.reshape(b, t, n, c)
        x = x + self.drop(ys)
        
        # Spatial FFN (MoE)
        xs_ffn = self.s_ffn_norm(x).reshape(b * t, n, c)
        ys_ffn = self.s_ffn(xs_ffn, is_temporal=False).reshape(b, t, n, c)
        x = x + self.drop(ys_ffn)
        
        # --- Temporal Attention ---
        xt = self.t_norm(x).transpose(1, 2).reshape(b * n, t, c)
        qkv = self.t_qkv(xt).reshape(b * n, t, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_rope:
            freqs_cis = self.freqs_cis[:t]
            q, k = apply_rotary_emb(q, k, freqs_cis)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        yt = F.scaled_dot_product_attention(q, k, v, dropout_p=self.t_attn_drop.p if self.training else 0.0)
        yt = yt.transpose(1, 2).reshape(b * n, t, c)
        yt = self.t_proj(yt)
        
        yt = yt.reshape(b, n, t, c).transpose(1, 2)
        x = x + self.drop(yt)
        
        # Temporal FFN (MoE with LTC)
        xt_ffn = self.t_ffn_norm(x).transpose(1, 2).reshape(b * n, t, c)
        yt_ffn = self.t_ffn(xt_ffn, is_temporal=True).reshape(b, n, t, c).transpose(1, 2)
        x = x + self.drop(yt_ffn)
        
        return x

class DSTFormerV3MoE(nn.Module):
    def __init__(self, cfg: DSTFormerV3MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_DSTFormerV3MoEBlock(cfg) for _ in range(int(cfg.num_layers))])
        self.out_norm = nn.LayerNorm(int(cfg.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)
