import torch
import torch.nn as nn
import torch.nn.functional as F

from .dstformer_v2 import apply_rotary_emb, precompute_freqs_cis
from .dstformer_v3_moe import SwiGLU

from dataclasses import dataclass
from typing import Optional

@dataclass
class DSTFormerV5GuidedMoEConfig:
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    num_layers: int = 4
    use_rope: bool = True
    use_ltc: bool = True
    num_experts: int = 8
    guide_mode: str = "none"  # "none", "bias", "cross_attn"

class DenseGuidedMoESwiGLU(nn.Module):
    """
    Guided Mixture of Experts (Dense MoE) over SwiGLU.
    Uses continuous codebook features to guide the router via bias or cross-attention.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, 
                 num_experts: int = 4, dropout: float = 0.0, use_ltc: bool = False,
                 guide_mode: str = "none"):
        super().__init__()
        self.num_experts = num_experts
        self.guide_mode = guide_mode
        
        # Base router from continuous feature
        self.router = nn.Linear(in_features, num_experts)
        
        if self.guide_mode == "bias":
            self.guide_proj = nn.Linear(in_features, num_experts)
        elif self.guide_mode == "cross_attn":
            self.guide_q = nn.Linear(in_features, in_features)
            self.expert_keys = nn.Parameter(torch.randn(num_experts, in_features))
            nn.init.normal_(self.expert_keys, std=0.02)
        
        self.experts = nn.ModuleList([
            SwiGLU(in_features, hidden_features, out_features, dropout, use_ltc)
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None, is_temporal: bool = False) -> torch.Tensor:
        # x: (B*T, N, C) if not is_temporal else (B*N, T, C)
        router_logits = self.router(x)  # (..., E)
        
        if guide is not None and self.guide_mode != "none":
            B, T, C = guide.shape
            
            if not is_temporal:
                # Spatial: x is (B*T, N, C)
                g = guide.reshape(B*T, 1, C)  # (B*T, 1, C)
            else:
                # Temporal: x is (B*N, T, C)
                N = x.shape[0] // B
                g = guide.unsqueeze(1).expand(B, N, T, C).reshape(B*N, T, C)
                
            if self.guide_mode == "bias":
                bias = self.guide_proj(g)  # (..., E)
                router_logits = router_logits + bias
            elif self.guide_mode == "cross_attn":
                q = self.guide_q(g)  # (..., C)
                attn_logits = torch.matmul(q, self.expert_keys.T) / (C ** 0.5)
                router_logits = router_logits + attn_logits
                
        routing_weights = F.softmax(router_logits, dim=-1)
        
        out = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x, is_temporal=is_temporal)
            out = out + expert_out * routing_weights[..., i:i+1]
            
        return out

class _DSTFormerV5GuidedMoEBlock(nn.Module):
    def __init__(self, cfg: DSTFormerV5GuidedMoEConfig, max_t: int = 1024):
        super().__init__()
        dim = int(cfg.dim)
        self.num_heads = int(cfg.num_heads)
        self.head_dim = dim // self.num_heads
        
        swiglu_hidden = int(dim * float(cfg.mlp_ratio) * 2 / 3)

        # ================= ST Branch (Spatial -> Temporal) =================
        self.st_s_norm = nn.LayerNorm(dim)
        self.st_s_attn = nn.MultiheadAttention(dim, self.num_heads, dropout=float(cfg.attn_dropout), batch_first=True)
        self.st_s_ffn_norm = nn.LayerNorm(dim)
        self.st_s_ffn = DenseGuidedMoESwiGLU(dim, swiglu_hidden, dim, cfg.num_experts, cfg.dropout, use_ltc=False, guide_mode=cfg.guide_mode)
        
        self.st_t_norm = nn.LayerNorm(dim)
        self.st_t_qkv = nn.Linear(dim, dim * 3)
        self.st_t_proj = nn.Linear(dim, dim)
        self.st_t_attn_drop = nn.Dropout(float(cfg.attn_dropout))
        self.st_t_ffn_norm = nn.LayerNorm(dim)
        self.st_t_ffn = DenseGuidedMoESwiGLU(dim, swiglu_hidden, dim, cfg.num_experts, cfg.dropout, use_ltc=cfg.use_ltc, guide_mode=cfg.guide_mode)

        # ================= TS Branch (Temporal -> Spatial) =================
        self.ts_t_norm = nn.LayerNorm(dim)
        self.ts_t_qkv = nn.Linear(dim, dim * 3)
        self.ts_t_proj = nn.Linear(dim, dim)
        self.ts_t_attn_drop = nn.Dropout(float(cfg.attn_dropout))
        self.ts_t_ffn_norm = nn.LayerNorm(dim)
        self.ts_t_ffn = DenseGuidedMoESwiGLU(dim, swiglu_hidden, dim, cfg.num_experts, cfg.dropout, use_ltc=cfg.use_ltc, guide_mode=cfg.guide_mode)
        
        self.ts_s_norm = nn.LayerNorm(dim)
        self.ts_s_attn = nn.MultiheadAttention(dim, self.num_heads, dropout=float(cfg.attn_dropout), batch_first=True)
        self.ts_s_ffn_norm = nn.LayerNorm(dim)
        self.ts_s_ffn = DenseGuidedMoESwiGLU(dim, swiglu_hidden, dim, cfg.num_experts, cfg.dropout, use_ltc=False, guide_mode=cfg.guide_mode)

        # ================= Fusion =================
        self.att_fuse = nn.Linear(dim * 2, 2)
        
        self.drop = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()
        self.use_rope = cfg.use_rope
        if self.use_rope:
            freqs_cis = precompute_freqs_cis(self.head_dim, max_t)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, n, c = x.shape
        
        # ================= ST Branch =================
        x_st = x
        # 1. Spatial
        xs = self.st_s_norm(x_st).reshape(b * t, n, c)
        ys, _ = self.st_s_attn(query=xs, key=xs, value=xs, need_weights=False)
        x_st = x_st + self.drop(ys.reshape(b, t, n, c))
        
        xs_ffn = self.st_s_ffn_norm(x_st).reshape(b * t, n, c)
        ys_ffn = self.st_s_ffn(xs_ffn, guide=guide, is_temporal=False)
        x_st = x_st + self.drop(ys_ffn.reshape(b, t, n, c))
        
        # 2. Temporal
        xt = self.st_t_norm(x_st).transpose(1, 2).reshape(b * n, t, c)
        qkv = self.st_t_qkv(xt).reshape(b * n, t, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, self.freqs_cis[:t])
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        yt = F.scaled_dot_product_attention(q, k, v, dropout_p=self.st_t_attn_drop.p if self.training else 0.0)
        yt = self.st_t_proj(yt.transpose(1, 2).reshape(b * n, t, c))
        x_st = x_st + self.drop(yt.reshape(b, n, t, c).transpose(1, 2))
        
        xt_ffn = self.st_t_ffn_norm(x_st).transpose(1, 2).reshape(b * n, t, c)
        yt_ffn = self.st_t_ffn(xt_ffn, guide=guide, is_temporal=True)
        x_st = x_st + self.drop(yt_ffn.reshape(b, n, t, c).transpose(1, 2))
        
        # ================= TS Branch =================
        x_ts = x
        # 1. Temporal
        xt2 = self.ts_t_norm(x_ts).transpose(1, 2).reshape(b * n, t, c)
        qkv2 = self.ts_t_qkv(xt2).reshape(b * n, t, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        if self.use_rope:
            q2, k2 = apply_rotary_emb(q2, k2, self.freqs_cis[:t])
        q2, k2, v2 = q2.transpose(1, 2), k2.transpose(1, 2), v2.transpose(1, 2)
        yt2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=self.ts_t_attn_drop.p if self.training else 0.0)
        yt2 = self.ts_t_proj(yt2.transpose(1, 2).reshape(b * n, t, c))
        x_ts = x_ts + self.drop(yt2.reshape(b, n, t, c).transpose(1, 2))
        
        xt2_ffn = self.ts_t_ffn_norm(x_ts).transpose(1, 2).reshape(b * n, t, c)
        yt2_ffn = self.ts_t_ffn(xt2_ffn, guide=guide, is_temporal=True)
        x_ts = x_ts + self.drop(yt2_ffn.reshape(b, n, t, c).transpose(1, 2))
        
        # 2. Spatial
        xs2 = self.ts_s_norm(x_ts).reshape(b * t, n, c)
        ys2, _ = self.ts_s_attn(query=xs2, key=xs2, value=xs2, need_weights=False)
        x_ts = x_ts + self.drop(ys2.reshape(b, t, n, c))
        
        xs2_ffn = self.ts_s_ffn_norm(x_ts).reshape(b * t, n, c)
        ys2_ffn = self.ts_s_ffn(xs2_ffn, guide=guide, is_temporal=False)
        x_ts = x_ts + self.drop(ys2_ffn.reshape(b, t, n, c))
        
        # ================= Fusion =================
        alpha = torch.cat([x_st, x_ts], dim=-1)  # (B, T, N, 2C)
        alpha = self.att_fuse(alpha)             # (B, T, N, 2)
        alpha = alpha.softmax(dim=-1)
        x_out = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]
        
        return x_out

class DSTFormerV5GuidedMoE(nn.Module):
    def __init__(self, cfg: DSTFormerV5GuidedMoEConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_DSTFormerV5GuidedMoEBlock(cfg) for _ in range(int(cfg.num_layers))])
        self.out_norm = nn.LayerNorm(int(cfg.dim))

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, guide=guide)
        return self.out_norm(x)
