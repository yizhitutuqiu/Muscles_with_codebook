from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class DSTFormerV2Config:
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    num_layers: int = 4
    use_rope: bool = True
    use_ltc: bool = True  # Local Temporal Convolution

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key.
    q, k: (B, T, num_heads, head_dim)
    freqs_cis: (T, head_dim // 2, 2)
    """
    b, t, h, d = q.shape
    q_ = q.float().reshape(b, t, h, d // 2, 2)
    k_ = k.float().reshape(b, t, h, d // 2, 2)
    
    # freqs_cis: (T, d//2, 2) -> (1, T, 1, d//2, 2)
    freqs = freqs_cis.view(1, t, 1, d // 2, 2)
    
    # Complex multiplication
    q_out = torch.stack([
        q_[..., 0] * freqs[..., 0] - q_[..., 1] * freqs[..., 1],
        q_[..., 0] * freqs[..., 1] + q_[..., 1] * freqs[..., 0]
    ], dim=-1).reshape(b, t, h, d)
    
    k_out = torch.stack([
        k_[..., 0] * freqs[..., 0] - k_[..., 1] * freqs[..., 1],
        k_[..., 0] * freqs[..., 1] + k_[..., 1] * freqs[..., 0]
    ], dim=-1).reshape(b, t, h, d)
    
    return q_out.type_as(q), k_out.type_as(k)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cis

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0, use_ltc: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.use_ltc = use_ltc
        if use_ltc:
            # Depthwise 1D conv over time
            self.ltc = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
            
    def forward(self, x: torch.Tensor, is_temporal: bool = False) -> torch.Tensor:
        # x: (B, L, C)
        v1 = self.w1(x)
        if self.use_ltc and is_temporal:
            b, l, c = v1.shape
            v1 = v1.transpose(1, 2) # (B, C, L)
            v1 = self.ltc(v1)
            v1 = v1.transpose(1, 2) # (B, L, C)
        
        hidden = F.silu(v1) * self.w2(x)
        return self.drop(self.w3(hidden))

class _DSTFormerV2Block(nn.Module):
    def __init__(self, cfg: DSTFormerV2Config, max_t: int = 1024):
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
        # Use 2/3 hidden for SwiGLU to keep parameters similar
        swiglu_hidden = int(hidden * 2 / 3)
        self.s_ffn = SwiGLU(dim, swiglu_hidden, dim, cfg.dropout, use_ltc=False)
        
        # Temporal
        self.t_norm = nn.LayerNorm(dim)
        self.t_qkv = nn.Linear(dim, dim * 3)
        self.t_proj = nn.Linear(dim, dim)
        self.t_attn_drop = nn.Dropout(float(cfg.attn_dropout))
        
        self.t_ffn_norm = nn.LayerNorm(dim)
        self.t_ffn = SwiGLU(dim, swiglu_hidden, dim, cfg.dropout, use_ltc=cfg.use_ltc)
        
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
        
        # Spatial FFN
        xs_ffn = self.s_ffn_norm(x).reshape(b * t, n, c)
        ys_ffn = self.s_ffn(xs_ffn, is_temporal=False).reshape(b, t, n, c)
        x = x + self.drop(ys_ffn)
        
        # --- Temporal Attention ---
        # Reshape to (B*N, T, C)
        xt = self.t_norm(x).transpose(1, 2).reshape(b * n, t, c)
        
        qkv = self.t_qkv(xt).reshape(b * n, t, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Each is (B*N, T, H, D)
        
        if self.use_rope:
            freqs_cis = self.freqs_cis[:t] # (T, D//2, 2)
            q, k = apply_rotary_emb(q, k, freqs_cis)
            
        q = q.transpose(1, 2) # (B*N, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        yt = F.scaled_dot_product_attention(q, k, v, dropout_p=self.t_attn_drop.p if self.training else 0.0)
        yt = yt.transpose(1, 2).reshape(b * n, t, c)
        yt = self.t_proj(yt)
        
        yt = yt.reshape(b, n, t, c).transpose(1, 2)
        x = x + self.drop(yt)
        
        # Temporal FFN with LTC
        xt_ffn = self.t_ffn_norm(x).transpose(1, 2).reshape(b * n, t, c)
        yt_ffn = self.t_ffn(xt_ffn, is_temporal=True).reshape(b, n, t, c).transpose(1, 2)
        x = x + self.drop(yt_ffn)
        
        return x

class DSTFormerV2(nn.Module):
    def __init__(self, cfg: DSTFormerV2Config):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_DSTFormerV2Block(cfg) for _ in range(int(cfg.num_layers))])
        self.out_norm = nn.LayerNorm(int(cfg.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)
