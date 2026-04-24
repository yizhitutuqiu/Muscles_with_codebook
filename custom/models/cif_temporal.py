import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal import TemporalConfig, _infer_seq_and_frame_dim, _TemporalBlock

class CIFTemporalEncoder(nn.Module):
    """
    CIF (Continuous Integrate-and-Fire) Encoder.
    Learns to dynamically segment continuous time-series (e.g. EMG) into `token_count` discrete semantic tokens.
    """
    def __init__(self, in_dim: int, hidden_dim: int, token_count: int, code_dim: int, temporal: TemporalConfig):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.token_count = token_count
        self.code_dim = code_dim
        
        self.seq_len, self.frame_dim = _infer_seq_and_frame_dim(in_dim=self.in_dim, temporal=temporal)
        
        # Backbone TCN
        k = int(temporal.kernel_size)
        layers = int(temporal.num_layers)
        drop = float(temporal.dropout)
        
        self.in_proj = nn.Conv1d(self.frame_dim, self.hidden_dim, kernel_size=1)
        blocks = []
        for i in range(max(1, layers)):
            blocks.append(_TemporalBlock(channels=self.hidden_dim, kernel_size=k, dilation=1, dropout=drop, use_residual=False))
        self.blocks = nn.Sequential(*blocks)
        
        self.to_code = nn.Conv1d(self.hidden_dim, self.code_dim, kernel_size=1)
        
        # CIF Weight Predictor
        self.weight_proj = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len * frame_dim)
        n = x.shape[0]
        xt = x.view(n, self.seq_len, self.frame_dim).transpose(1, 2).contiguous() # (B, C, L)
        
        h = self.in_proj(xt)
        h = self.blocks(h) # (B, hidden_dim, L)
        
        y = self.to_code(h).transpose(1, 2) # (B, L, code_dim)
        
        # Predict alpha (B, L)
        alpha = self.weight_proj(h).squeeze(1) # (B, L)
        
        # Scale alpha so that sum equals exactly token_count
        alpha_sum = alpha.sum(dim=1, keepdim=True) + 1e-8
        alpha_scaled = alpha * (self.token_count / alpha_sum) # (B, L)
        
        # Vectorized CIF Integration
        A_end = torch.cumsum(alpha_scaled, dim=1) # (B, L)
        A_start = A_end - alpha_scaled # (B, L)
        
        k_start = torch.arange(self.token_count, device=x.device, dtype=x.dtype).view(1, 1, self.token_count)
        k_end = k_start + 1.0
        
        A_start_exp = A_start.unsqueeze(2) # (B, L, 1)
        A_end_exp = A_end.unsqueeze(2) # (B, L, 1)
        
        inter_start = torch.max(A_start_exp, k_start) # (B, L, K)
        inter_end = torch.min(A_end_exp, k_end) # (B, L, K)
        
        weight = torch.clamp(inter_end - inter_start, min=0.0) # (B, L, K)
        
        # CIF output: (B, K, code_dim)
        out = torch.bmm(weight.transpose(1, 2), y) # (B, K, code_dim)
        
        # Flatten for VQ interface
        return out.view(n, self.token_count * self.code_dim)

class CIFTemporalDecoder(nn.Module):
    """
    CIF Decoder (De-CIF).
    Takes K tokens and upsamples them back to L frames using standard linear interpolation or transposed conv.
    Since we don't save the exact alpha_scaled from encoder during VQ training easily without API change,
    we use a standard temporal upsampler to invert the process. 
    A 1D interpolation works well.
    """
    def __init__(self, out_dim: int, token_count: int, code_dim: int, temporal: TemporalConfig):
        super().__init__()
        self.out_dim = out_dim
        self.token_count = token_count
        self.code_dim = code_dim
        
        self.seq_len, self.frame_dim = _infer_seq_and_frame_dim(in_dim=self.out_dim, temporal=temporal)
        self.to_frame = nn.Conv1d(self.code_dim, self.frame_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, token_count * code_dim)
        n = z.shape[0]
        y = z.view(n, self.token_count, self.code_dim).transpose(1, 2).contiguous() # (B, code_dim, K)
        
        # Upsample K -> L
        y = F.interpolate(y, size=self.seq_len, mode="linear", align_corners=False) # (B, code_dim, L)
        y = self.to_frame(y) # (B, frame_dim, L)
        
        return y.transpose(1, 2).contiguous().view(n, self.seq_len * self.frame_dim)
