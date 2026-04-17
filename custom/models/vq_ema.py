from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VQEMAConfig:
    num_codes: int
    code_dim: int
    decay: float = 0.99
    eps: float = 1e-5
    beta: float = 10.0

    # Prevent unused codes from decaying to zero (preserve k-means init / avoid collapse)
    min_cluster_size_for_update: float = 0.01  # codes with ema_cluster_size < this keep current embedding

    # dead-code reset
    reset_dead_codes: bool = True
    dead_code_threshold: float = 1e-4
    reset_interval: int = 200
    reset_samples: int = 2048


class VectorQuantizerEMA(nn.Module):
    """
    EMA-updated vector quantizer with optional dead-code reset.

    Input:
      z_e: (N, D)
    Output:
      z_q_st: straight-through quantized vectors, shape (N, D)
      indices: code ids, shape (N,)
      vq_loss: commitment loss
      perplexity: scalar
      avg_probs: batch usage probability over codes, shape (K,)
    """

    def __init__(self, cfg: VQEMAConfig):
        super().__init__()
        self.cfg = cfg
        k = int(cfg.num_codes)
        d = int(cfg.code_dim)

        # Encoder output (after BN + to_code) has per-dim variance ~1, so ||z_e|| ~ sqrt(d).
        # Initialize codebook so ||e|| ~ sqrt(d) to match: use randn without 1/sqrt(d).
        embed = torch.randn(k, d)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(k))
        self.register_buffer("ema_embed_sum", embed.clone())
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("num_resets", torch.tensor(0, dtype=torch.long))

    def _dist(self, z: torch.Tensor) -> torch.Tensor:
        z2 = (z * z).sum(dim=1, keepdim=True)
        e2 = (self.embedding * self.embedding).sum(dim=1).view(1, -1)
        ze = z @ self.embedding.t()
        return z2 + e2 - 2.0 * ze

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, onehot: torch.Tensor) -> None:
        decay = float(self.cfg.decay)
        eps = float(self.cfg.eps)

        cluster_size = onehot.sum(dim=0)
        
        # Optimize embed_sum memory for large batches
        # instead of full matrix multiply `onehot.t() @ z` which creates large temporary variables
        # use a more memory efficient approach if possible, or just standard matmul
        embed_sum = torch.matmul(onehot.t(), z)

        self.ema_cluster_size.mul_(decay).add_(cluster_size, alpha=1.0 - decay)
        self.ema_embed_sum.mul_(decay).add_(embed_sum, alpha=1.0 - decay)

        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + eps) / (n + self.cfg.num_codes * eps) * n
        new_embedding = self.ema_embed_sum / smoothed.unsqueeze(1).clamp_min(eps)
        # Keep current embedding for codes that are effectively unused (avoid decay to zero -> collapse)
        thresh = float(self.cfg.min_cluster_size_for_update)
        if thresh > 0:
            keep = self.ema_cluster_size < thresh
            if keep.any():
                self.embedding.copy_(torch.where(keep.unsqueeze(1), self.embedding, new_embedding))
                return
        self.embedding.copy_(new_embedding)

    @torch.no_grad()
    def _maybe_reset_dead_codes(self, z: torch.Tensor, usage_prob: torch.Tensor) -> None:
        if not bool(self.cfg.reset_dead_codes):
            return
        step = int(self.step.item())
        if step <= 0 or (step % int(self.cfg.reset_interval) != 0):
            return

        dead = usage_prob < float(self.cfg.dead_code_threshold)
        num_dead = int(dead.sum().item())
        if num_dead <= 0:
            return
        if z.shape[0] <= 0:
            return

        m = min(int(self.cfg.reset_samples), int(z.shape[0]))
        perm = torch.randperm(int(z.shape[0]), device=z.device)[:m]
        pool = z[perm].detach()
        # Prefer sampling without replacement to avoid duplicate codes (reduces collapse risk)
        if num_dead <= m:
            pick = torch.randperm(m, device=z.device)[:num_dead]
            new_vecs = pool[pick].clone()
        else:
            choose = torch.randint(low=0, high=m, size=(num_dead,), device=z.device)
            new_vecs = pool[choose].clone()
        # Small jitter so reset codes stay distinct when pool has few unique vectors
        new_vecs = new_vecs + torch.randn_like(new_vecs, device=z.device) * (z.std().clamp_min(1e-6) * 1e-2 + 1e-6)
        dead_idx = torch.nonzero(dead, as_tuple=False).flatten()

        self.embedding[dead_idx] = new_vecs
        self.ema_embed_sum[dead_idx] = new_vecs
        self.ema_cluster_size[dead_idx] = 1.0
        self.num_resets.add_(num_dead)

    def forward(
        self,
        z_e: torch.Tensor,
        *,
        skip_dead_code_reset: bool = False,
        explore_noise_std_frac: float = 0.0,
        commitment_beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_e.ndim != 2 or z_e.shape[1] != int(self.cfg.code_dim):
            raise ValueError(f"Expected z_e shape (N,{self.cfg.code_dim}), got {tuple(z_e.shape)}")

        # Optional ε-greedy-style exploration: noisy z only for nearest-code assignment (not for EMA sums).
        if self.training and float(explore_noise_std_frac) > 0:
            sigma = z_e.std().clamp_min(1e-6) * float(explore_noise_std_frac)
            z_assign = z_e + torch.randn_like(z_e) * sigma
        else:
            z_assign = z_e

        # Optimize dist computation to avoid huge memory spikes and slow computation when batch_size or num_codes is very large
        # z_assign: (N, D), embedding: (K, D)
        # Using memory efficient implementation:
        dist = torch.cdist(z_assign, self.embedding, p=2.0) ** 2

        indices = torch.argmin(dist, dim=1)
        onehot = F.one_hot(indices, num_classes=int(self.cfg.num_codes)).type_as(z_e)

        z_q = onehot @ self.embedding

        if self.training:
            self._ema_update(z_e, onehot)

        avg_probs = onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training:
            self.step.add_(1)
            if not skip_dead_code_reset:
                self._maybe_reset_dead_codes(z_e, avg_probs)

        # Standard VQ-VAE style commitment term:
        #   beta * || f_i - sg[c_q(f_i)] ||^2
        # We use the mean over all tokens/features for stable optimization.
        commit = F.mse_loss(z_e, z_q.detach())
        b = float(self.cfg.beta) if commitment_beta is None else float(commitment_beta)
        vq_loss = b * commit
        # STE: forward 时取值必须为 z_q（离散），梯度沿 z_e 传回。公式 z_e + (z_q - z_e).detach() 的 forward 值 = z_q。
        # 若错写成 z_q + (z_e - z_q).detach() 则 forward 值 = z_e，Decoder 会收到连续特征，train loss 可虚假到 1e-8。
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices, vq_loss, perplexity, avg_probs

