from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MoEFFNConfig:
    d_model: int = 128
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_experts: int = 4
    router_in_dim: int = 256


class MoEFeedForward(nn.Module):
    def __init__(self, cfg: MoEFFNConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        f = int(cfg.dim_feedforward)
        e = int(cfg.num_experts)
        self.router = nn.Linear(int(cfg.router_in_dim), e)
        self.drop = nn.Dropout(float(cfg.dropout))
        self.fc1 = nn.ModuleList([nn.Linear(d, f) for _ in range(e)])
        self.fc2 = nn.ModuleList([nn.Linear(f, d) for _ in range(e)])
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, *, router_x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x (T,B,C), got {tuple(x.shape)}")
        if router_x.ndim != 3:
            raise ValueError(f"Expected router_x (T,B,R), got {tuple(router_x.shape)}")
        if x.shape[0] != router_x.shape[0] or x.shape[1] != router_x.shape[1]:
            raise ValueError(f"Shape mismatch x={tuple(x.shape)} router_x={tuple(router_x.shape)}")

        logits = self.router(router_x)
        idx = torch.argmax(logits, dim=-1)

        outs = []
        for e, (l1, l2) in enumerate(zip(self.fc1, self.fc2)):
            y = l2(self.drop(self.act(l1(x))))
            outs.append(y)
        y_all = torch.stack(outs, dim=0)

        gather_idx = idx.unsqueeze(0).unsqueeze(-1).expand(1, idx.shape[0], idx.shape[1], x.shape[2])
        return y_all.gather(0, gather_idx).squeeze(0)


class TransformerEncoderLayerMoE(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        nhead: int = 16,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_experts: int = 4,
        router_in_dim: int = 256,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoEFeedForward(
            MoEFFNConfig(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_experts=num_experts,
                router_in_dim=router_in_dim,
            )
        )

    def forward(
        self,
        src: torch.Tensor,
        *,
        router_src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, need_weights=False)
        src = self.norm1(src + self.drop1(attn_out))
        ffn_out = self.moe(src, router_x=router_src)
        src = self.norm2(src + self.drop2(ffn_out))
        return src

