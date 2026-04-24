from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPConfig, MLPMixerConfig, FourLayerMLPMixer, FourLayerMLPMixerDecoder, TwoLayerMLP
from .temporal import (
    TemporalConfig,
    TemporalConv1dEncoder,
    TemporalTCNEncoder,
    MixerTCNEncoder,
    TemporalConvDecoder,
)
from .vq_ema import VQEMAConfig, VectorQuantizerEMA
from ..utils.online_standardize import OnlineStandardizeConfig, OnlineStandardizer


@dataclass(frozen=True)
class ModalityConfig:
    name: str
    in_dim: int
    hidden_dim: int
    token_count: int
    code_dim: int
    encoder_type: str = "mixer"
    decoder_type: str = "mixer"
    recon_weight: float = 1.0
    online_std: bool = True
    std_cfg: OnlineStandardizeConfig = field(default_factory=OnlineStandardizeConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    mixer: Optional[Dict[str, Any]] = None  # used when encoder_type is mixer_tcn
    frame_mixer: Optional[Dict[str, Any]] = None  # used when encoder_type is frame_mixer: tokens_per_frame, etc.


@dataclass(frozen=True)
class FrameCodebookConfig:
    vq: VQEMAConfig
    joints3d: ModalityConfig
    smpl_pose: Optional[ModalityConfig]
    emg: ModalityConfig
    encoder_decoder_only: bool = False  # 若 True：完全不用 codebook，只做 encoder→decoder 自重建，无交叉重建


class PerFrameMixerEncoder(nn.Module):
    """
    每帧独立用同一个 Mixer 编码：输入 (B, num_frames*frame_in_dim)，逐帧 (B, frame_in_dim) → Mixer → (B, tokens_per_frame*code_dim)，
    输出 (B, num_frames*tokens_per_frame*code_dim)。要求 cfg.token_count == num_frames * tokens_per_frame。
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        num_frames: int,
        tokens_per_frame: int,
        code_dim: int,
        in_dim: int,
    ):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.num_frames = int(num_frames)
        self.tokens_per_frame = int(tokens_per_frame)
        self.code_dim = int(code_dim)
        self.frame_in_dim = int(in_dim) // int(num_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, self.num_frames, self.frame_in_dim)
        x = x.view(B * self.num_frames, self.frame_in_dim)
        z = self.frame_encoder(x)
        return z.view(B, self.num_frames * self.tokens_per_frame * self.code_dim)


class PerFrameMixerDecoder(nn.Module):
    """
    与 PerFrameMixerEncoder 对称：输入 (B, num_frames*tokens_per_frame*code_dim)，逐帧 (B, tokens_per_frame*code_dim) → Mixer Decoder → (B, frame_out_dim)，
    输出 (B, num_frames*frame_out_dim)。
    """

    def __init__(
        self,
        frame_decoder: nn.Module,
        num_frames: int,
        tokens_per_frame: int,
        code_dim: int,
        frame_out_dim: int,
    ):
        super().__init__()
        self.frame_decoder = frame_decoder
        self.num_frames = int(num_frames)
        self.tokens_per_frame = int(tokens_per_frame)
        self.code_dim = int(code_dim)
        self.frame_out_dim = int(frame_out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z = z.view(B, self.num_frames, self.tokens_per_frame * self.code_dim)
        z = z.view(B * self.num_frames, self.tokens_per_frame * self.code_dim)
        out = self.frame_decoder(z)
        return out.view(B, self.num_frames * self.frame_out_dim)


class _ModalityAE(nn.Module):
    def __init__(self, cfg: ModalityConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = int(cfg.token_count) * int(cfg.code_dim)
        mixer_cfg = MLPMixerConfig(
            in_dim=cfg.in_dim,
            token_count=cfg.token_count,
            code_dim=cfg.code_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=4,
        )
        self.encoder = self._build_encoder(cfg, mixer_cfg)
        self.decoder = self._build_decoder(cfg, mixer_cfg)
        self.standardizer = OnlineStandardizer(cfg.in_dim, cfg=cfg.std_cfg) if cfg.online_std else None

    def _build_encoder(self, cfg: ModalityConfig, mixer_cfg: MLPMixerConfig) -> nn.Module:
        encoder_type = str(cfg.encoder_type).lower().strip()
        if encoder_type in ("mixer", "mlp_mixer", "four_layer_mlp_mixer"):
            return FourLayerMLPMixer(mixer_cfg)
        if encoder_type == "frame_mixer":
            fm = cfg.frame_mixer or {}
            num_frames = int(cfg.temporal.seq_len or (cfg.in_dim // int(cfg.temporal.frame_dim or 1)))
            frame_in_dim = int(cfg.temporal.frame_dim) if cfg.temporal.frame_dim is not None else (cfg.in_dim // num_frames)
            tokens_per_frame = int(fm.get("tokens_per_frame", 63))
            if num_frames * tokens_per_frame != int(cfg.token_count):
                raise ValueError(
                    f"frame_mixer: num_frames*tokens_per_frame={num_frames}*{tokens_per_frame} != token_count={cfg.token_count}"
                )
            frame_mixer_cfg = MLPMixerConfig(
                in_dim=frame_in_dim,
                token_count=tokens_per_frame,
                code_dim=cfg.code_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=int(fm.get("num_layers", 4)),
                token_mlp_dim=int(fm["token_mlp_dim"]) if fm.get("token_mlp_dim") is not None else None,
                channel_mlp_dim=int(fm["channel_mlp_dim"]) if fm.get("channel_mlp_dim") is not None else None,
                act=str(fm.get("act", "gelu")),
                dropout=float(fm.get("dropout", 0.0)),
            )
            frame_enc = FourLayerMLPMixer(frame_mixer_cfg)
            return PerFrameMixerEncoder(
                frame_encoder=frame_enc,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                code_dim=cfg.code_dim,
                in_dim=cfg.in_dim,
            )
        if encoder_type in ("mlp", "two_layer_mlp"):
            return TwoLayerMLP(MLPConfig(cfg.in_dim, cfg.hidden_dim, self.latent_dim))
        if encoder_type in ("conv1d", "temporal_conv1d"):
            return TemporalConv1dEncoder(
                in_dim=int(cfg.in_dim),
                hidden_dim=int(cfg.hidden_dim),
                token_count=int(cfg.token_count),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        if encoder_type in ("tcn", "temporal_tcn"):
            from .temporal import TemporalTCNEncoder
            return TemporalTCNEncoder(
                in_dim=int(cfg.in_dim),
                hidden_dim=int(cfg.hidden_dim),
                token_count=int(cfg.token_count),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        if encoder_type in ("clip_unified_tcn", "unified_tcn"):
            from .temporal import ClipUnifiedTCNEncoder
            return ClipUnifiedTCNEncoder(
                in_dim=int(cfg.in_dim),
                hidden_dim=int(cfg.hidden_dim),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        if encoder_type in ("cif", "temporal_cif"):
            from .cif_temporal import CIFTemporalEncoder
            return CIFTemporalEncoder(
                in_dim=int(cfg.in_dim),
                hidden_dim=int(cfg.hidden_dim),
                token_count=int(cfg.token_count),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        if encoder_type in ("mixer_tcn", "mixer+tcn"):
            mix_d = cfg.mixer or {}
            mix_tc = int(mix_d.get("token_count", cfg.token_count))
            mix_cd = int(mix_d.get("code_dim", 256))
            mix_h = int(mix_d.get("hidden_dim", 512))
            mix_layers = int(mix_d.get("num_layers", 4))
            mix_token_mlp = mix_d.get("token_mlp_dim")
            mix_channel_mlp = mix_d.get("channel_mlp_dim")
            mix_act = str(mix_d.get("act", "gelu"))
            mix_drop = float(mix_d.get("dropout", 0.0))
            mixer_cfg = MLPMixerConfig(
                in_dim=int(cfg.in_dim),
                token_count=mix_tc,
                code_dim=mix_cd,
                hidden_dim=mix_h,
                num_layers=mix_layers,
                token_mlp_dim=int(mix_token_mlp) if mix_token_mlp is not None else None,
                channel_mlp_dim=int(mix_channel_mlp) if mix_channel_mlp is not None else None,
                act=mix_act,
                dropout=mix_drop,
            )
            # TCN 的输入是 Mixer 输出 (mix_tc * mix_cd)，故 temporal 用 mixer 的 token 数/维，不是原始 seq_len
            temporal_for_tcn = TemporalConfig(
                seq_len=mix_tc,
                frame_dim=mix_cd,
                kernel_size=cfg.temporal.kernel_size,
                num_layers=cfg.temporal.num_layers,
                dilation_base=cfg.temporal.dilation_base,
                dropout=cfg.temporal.dropout,
                pool=cfg.temporal.pool,
                upsample=cfg.temporal.upsample,
            )
            return MixerTCNEncoder(
                in_dim=int(cfg.in_dim),
                mixer_cfg=mixer_cfg,
                tcn_hidden_dim=int(cfg.hidden_dim),
                token_count=int(cfg.token_count),
                code_dim=int(cfg.code_dim),
                temporal=temporal_for_tcn,
            )
        raise ValueError(f"Unsupported encoder_type for {cfg.name}: {cfg.encoder_type}")

    def _build_decoder(self, cfg: ModalityConfig, mixer_cfg: MLPMixerConfig) -> nn.Module:
        decoder_type = str(cfg.decoder_type).lower().strip()
        if decoder_type in ("mixer", "mlp_mixer", "four_layer_mlp_mixer"):
            return FourLayerMLPMixerDecoder(mixer_cfg)
        if decoder_type == "frame_mixer":
            fm = cfg.frame_mixer or {}
            num_frames = int(cfg.temporal.seq_len or (cfg.in_dim // int(cfg.temporal.frame_dim or 1)))
            frame_out_dim = int(cfg.temporal.frame_dim) if cfg.temporal.frame_dim is not None else (cfg.in_dim // num_frames)
            tokens_per_frame = int(fm.get("tokens_per_frame", 63))
            if num_frames * tokens_per_frame != int(cfg.token_count):
                raise ValueError(
                    f"frame_mixer decoder: num_frames*tokens_per_frame != token_count"
                )
            frame_decoder_cfg = MLPMixerConfig(
                in_dim=frame_out_dim,
                token_count=tokens_per_frame,
                code_dim=cfg.code_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=int(fm.get("num_layers", 4)),
                token_mlp_dim=int(fm["token_mlp_dim"]) if fm.get("token_mlp_dim") is not None else None,
                channel_mlp_dim=int(fm["channel_mlp_dim"]) if fm.get("channel_mlp_dim") is not None else None,
                act=str(fm.get("act", "gelu")),
                dropout=float(fm.get("dropout", 0.0)),
            )
            frame_dec = FourLayerMLPMixerDecoder(frame_decoder_cfg)
            return PerFrameMixerDecoder(
                frame_decoder=frame_dec,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                code_dim=cfg.code_dim,
                frame_out_dim=frame_out_dim,
            )
        if decoder_type in ("mlp", "two_layer_mlp"):
            return TwoLayerMLP(MLPConfig(self.latent_dim, cfg.hidden_dim, cfg.in_dim))
        if decoder_type in ("conv1d", "tcn", "temporal_conv1d", "temporal_tcn"):
            # Decoder shape is identical for conv1d vs TCN variants: token sequence -> upsample -> per-timestep projection.
            from .temporal import TemporalConvDecoder
            return TemporalConvDecoder(
                out_dim=int(cfg.in_dim),
                token_count=int(cfg.token_count),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        if decoder_type in ("clip_unified_tcn", "unified_tcn"):
            from .temporal import ClipUnifiedTCNDecoder
            return ClipUnifiedTCNDecoder(
                out_dim=int(cfg.in_dim),
                hidden_dim=int(cfg.hidden_dim),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        if decoder_type in ("cif", "temporal_cif"):
            from .cif_temporal import CIFTemporalDecoder
            return CIFTemporalDecoder(
                out_dim=int(cfg.in_dim),
                token_count=int(cfg.token_count),
                code_dim=int(cfg.code_dim),
                temporal=cfg.temporal,
            )
        raise ValueError(f"Unsupported decoder_type for {cfg.name}: {cfg.decoder_type}")

    def normalize(self, x: torch.Tensor, *, update: bool) -> torch.Tensor:
        if self.standardizer is None:
            return x
        return self.standardizer(x, update=update)


class FrameCodebookModel(nn.Module):
    """
    Shared-codebook frame-wise VQ model for 3D joints and EMG.

    Training objective:
    - 3D -> 3D self reconstruction
    - EMG -> EMG self reconstruction
    - 3D -> EMG cross reconstruction
    - EMG -> 3D cross reconstruction
    - plus standard VQ commitment term with beta from cfg.vq

    Reconstruction loss is SmoothL1 in the normalized feature space.
    No extra alignment / contrastive / auxiliary losses are used.
    """

    def __init__(self, cfg: FrameCodebookConfig):
        super().__init__()
        self.cfg = cfg

        code_dim = int(cfg.vq.code_dim)
        base_tokens = int(cfg.joints3d.token_count)
        for mod in (cfg.joints3d, cfg.smpl_pose, cfg.emg):
            if mod is None:
                continue
            if int(mod.code_dim) != code_dim:
                raise ValueError(f"{mod.name}.code_dim={mod.code_dim} != vq.code_dim={code_dim}")
            if int(mod.token_count) <= 0:
                raise ValueError(f"{mod.name}.token_count must be > 0, got {mod.token_count}")
            # Cross reconstruction reuses quantized tokens across modalities, so token_count must match.
            if int(mod.token_count) != base_tokens:
                raise ValueError(
                    f"{mod.name}.token_count={mod.token_count} != joints3d.token_count={base_tokens} "
                    "but cross reconstruction assumes the same token_count across modalities."
                )

        self.vq = VectorQuantizerEMA(cfg.vq)
        self.joints3d = _ModalityAE(cfg.joints3d)
        self.smpl_pose = _ModalityAE(cfg.smpl_pose) if cfg.smpl_pose is not None else None
        self.emg = _ModalityAE(cfg.emg)

    def _encode_quantize(
        self,
        mod: _ModalityAE,
        x: torch.Tensor,
        *,
        bypass_vq: bool = False,
        skip_vq_dead_reset: bool = False,
        vq_explore_noise_std_frac: float = 0.0,
        vq_commitment_beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.float()
        x_n = mod.normalize(x, update=self.training)
        batch_size = int(x_n.shape[0])
        token_count = int(mod.cfg.token_count)
        code_dim = int(mod.cfg.code_dim)

        # (B, token_count * code_dim) -> (B, token_count, code_dim)
        z_e = mod.encoder(x_n).view(batch_size, token_count, code_dim)

        if bypass_vq:
            # Cold-start warmup: encoder -> decoder only, codebook not used or updated.
            # 此时 Decoder 收到的是 z_e（连续），不是 z_q，故 recon loss 可低至 1e-8；与 eval（始终用 z_q）不可直接对比。
            z_q = z_e.view(batch_size, token_count * code_dim)
            n_tokens = batch_size * token_count
            dev = z_e.device
            indices = torch.zeros(n_tokens, dtype=torch.long, device=dev)
            vq_loss = torch.tensor(0.0, device=dev)
            perplexity = torch.tensor(0.0, device=dev)
            usage = torch.tensor(0.0, device=dev)
            return x_n, z_q, indices, vq_loss, perplexity, usage

        # Quantize each token independently in the SAME shared codebook.
        z_q_tokens, indices, vq_loss, perplexity, usage = self.vq(
            z_e.reshape(batch_size * token_count, code_dim),
            skip_dead_code_reset=skip_vq_dead_reset,
            explore_noise_std_frac=vq_explore_noise_std_frac,
            commitment_beta=vq_commitment_beta,
        )
        z_q = z_q_tokens.view(batch_size, token_count * code_dim)
        return x_n, z_q, indices, vq_loss, perplexity, usage

    @torch.no_grad()
    def get_encoder_outputs_for_init(
        self,
        *,
        x_joints3d: torch.Tensor,
        x_smpl_pose: Optional[torch.Tensor] = None,
        x_emg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run encoders only and return z_e per modality for codebook k-means init.
        Returns dict modality_name -> z_e of shape (B, token_count, code_dim).
        """
        out: Dict[str, torch.Tensor] = {}
        x_joints3d = x_joints3d.float()
        xj_n = self.joints3d.normalize(x_joints3d, update=False)
        b = int(xj_n.shape[0])
        tc = int(self.joints3d.cfg.token_count)
        cd = int(self.joints3d.cfg.code_dim)
        z_j = self.joints3d.encoder(xj_n).view(b, tc, cd)
        out["joints3d"] = z_j
        if self.emg is not None and x_emg.numel() > 0:
            x_emg = x_emg.float()
            xe_n = self.emg.normalize(x_emg, update=False)
            be = int(xe_n.shape[0])
            te = int(self.emg.cfg.token_count)
            ce = int(self.emg.cfg.code_dim)
            z_e = self.emg.encoder(xe_n).view(be, te, ce)
            out["emg"] = z_e
        if self.smpl_pose is not None and x_smpl_pose is not None and x_smpl_pose.numel() > 0:
            x_smpl_pose = x_smpl_pose.float()
            xs_n = self.smpl_pose.normalize(x_smpl_pose, update=False)
            bs = int(xs_n.shape[0])
            ts = int(self.smpl_pose.cfg.token_count)
            cs = int(self.smpl_pose.cfg.code_dim)
            z_s = self.smpl_pose.encoder(xs_n).view(bs, ts, cs)
            out["smpl_pose"] = z_s
        return out

    def _decode(self, mod: _ModalityAE, z_q: torch.Tensor) -> torch.Tensor:
        return mod.decoder(z_q)

    def _smooth_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target)

    def _record_codebook_stats(
        self,
        *,
        name: str,
        indices: torch.Tensor,
        vq_loss: torch.Tensor,
        perplexity: torch.Tensor,
        usage: torch.Tensor,
        token_count: int,
        stats: Dict[str, torch.Tensor],
    ) -> None:
        stats[f"{name}/vq_loss"] = vq_loss.detach()
        stats[f"{name}/perplexity"] = perplexity.detach()
        stats[f"{name}/usage_mean"] = usage.detach().mean()
        stats[f"{name}/usage_min"] = usage.detach().min()
        stats[f"{name}/usage_max"] = usage.detach().max()
        stats[f"{name}/codes_unique"] = torch.tensor(torch.unique(indices).numel(), device=indices.device)
        stats[f"{name}/token_count"] = torch.tensor(token_count, device=indices.device)

    def forward(
        self,
        *,
        x_joints3d: torch.Tensor,
        x_smpl_pose: Optional[torch.Tensor] = None,
        x_emg: torch.Tensor,
        emg_weight: float = 1.0,
        bypass_vq: bool = False,
        skip_vq_dead_reset: bool = False,
        vq_explore_noise_std_frac: float = 0.0,
        vq_commitment_beta: Optional[float] = None,
        loss_w_joints3d_self: float = 1.0,
        loss_w_vq_joints3d: float = 1.0,
        loss_w_emg_self: float = 1.0,
        loss_w_emg_to_joints3d: float = 1.0,
        loss_w_joints3d_to_emg: float = 1.0,
        loss_w_vq_emg: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param emg_weight: Weight for EMG-related losses (0 = j3d-only pre-training).
        :param bypass_vq: If True, skip codebook (encoder output -> decoder directly, no VQ loss). For warmup.
        :param skip_vq_dead_reset: If True, do not replace dead codes (e.g. warmup_2 after k-means).
        :param vq_explore_noise_std_frac: Train-only: std_frac * std(z_e) Gaussian noise on z for argmin only.
        :param loss_w_*: Per-term weights (default 1.0). See model.loss_weights in YAML.
        """
        # 土拨鼠之日排查：确认训练脚本传入的 bypass_vq 是否真的生效（首步打一次即可）
        encoder_decoder_only = getattr(self.cfg, "encoder_decoder_only", False)
        effective_bypass = bypass_vq or encoder_decoder_only
        if self.training and not getattr(self, "_bypass_vq_debug_done", False):
            print(
                f"[DEBUG IN MODEL] bypass_vq={bypass_vq}, encoder_decoder_only={encoder_decoder_only} "
                f"→ effective_bypass={effective_bypass} (Decoder 输入=z_e，无 VQ；仅自重建)"
            )
            self._bypass_vq_debug_done = True
        stats: Dict[str, torch.Tensor] = {}

        xj_n, zq_j, idx_j, vq_j, ppl_j, use_j = self._encode_quantize(
            self.joints3d,
            x_joints3d,
            bypass_vq=effective_bypass,
            skip_vq_dead_reset=skip_vq_dead_reset,
            vq_explore_noise_std_frac=vq_explore_noise_std_frac,
            vq_commitment_beta=vq_commitment_beta,
        )

        self._record_codebook_stats(
            name="joints3d",
            indices=idx_j,
            vq_loss=vq_j,
            perplexity=ppl_j,
            usage=use_j,
            token_count=int(self.joints3d.cfg.token_count),
            stats=stats,
        )

        # Always optimize joints3d self-reconstruction + its VQ term.
        pred_j_from_j = self._decode(self.joints3d, zq_j)
        loss_j_self = self._smooth_l1(pred_j_from_j, xj_n)

        stats["joints3d/self_smooth_l1"] = loss_j_self.detach()

        # Keep backward-compatible aliases for logging convenience.
        stats["joints3d/recon_mse"] = loss_j_self.detach()
        w_j = float(loss_w_joints3d_self)
        w_vqj = float(loss_w_vq_joints3d)
        total = w_j * float(self.joints3d.cfg.recon_weight) * loss_j_self
        if not encoder_decoder_only and not bypass_vq:
            total = total + w_vqj * vq_j

        # IMPORTANT:
        # When emg_weight==0 (e.g. AMASS j3d-only pre-training), we must skip EMG encode/quantize
        # entirely. Otherwise the shared EMA VQ codebook will still be updated by EMG tokens
        # (AMASS provides all-zero EMG), which can collapse the codebook and harm joints3d.
        if float(emg_weight) != 0.0:
            xe_n, zq_e, idx_e, vq_e, ppl_e, use_e = self._encode_quantize(
                self.emg,
                x_emg,
                bypass_vq=effective_bypass,
                skip_vq_dead_reset=skip_vq_dead_reset,
                vq_explore_noise_std_frac=vq_explore_noise_std_frac,
                vq_commitment_beta=vq_commitment_beta,
            )
            self._record_codebook_stats(
                name="emg",
                indices=idx_e,
                vq_loss=vq_e,
                perplexity=ppl_e,
                usage=use_e,
                token_count=int(self.emg.cfg.token_count),
                stats=stats,
            )

            # Four reconstruction routes. Warmup (bypass_vq): only self-recon, no cross (no codebook).
            pred_e_from_e = self._decode(self.emg, zq_e)
            loss_e_self = self._smooth_l1(pred_e_from_e, xe_n)
            stats["emg/self_smooth_l1"] = loss_e_self.detach()
            stats["emg/recon_mse"] = loss_e_self.detach()

            w_es = float(loss_w_emg_self)
            w_etj = float(loss_w_emg_to_joints3d)
            w_jte = float(loss_w_joints3d_to_emg)
            w_vqe = float(loss_w_vq_emg)
            if effective_bypass:
                total = total + float(emg_weight) * w_es * float(self.emg.cfg.recon_weight) * loss_e_self
                stats["emg_to_joints3d/smooth_l1"] = torch.tensor(0.0, device=total.device)
                stats["joints3d_to_emg/smooth_l1"] = torch.tensor(0.0, device=total.device)
            else:
                pred_j_from_e = self._decode(self.joints3d, zq_e)
                pred_e_from_j = self._decode(self.emg, zq_j)
                loss_e_to_j = self._smooth_l1(pred_j_from_e, xj_n)
                loss_j_to_e = self._smooth_l1(pred_e_from_j, xe_n)
                stats["emg_to_joints3d/smooth_l1"] = loss_e_to_j.detach()
                stats["joints3d_to_emg/smooth_l1"] = loss_j_to_e.detach()
                total = total + float(emg_weight) * (
                    w_es * float(self.emg.cfg.recon_weight) * loss_e_self
                    + w_etj * loss_e_to_j
                    + w_jte * loss_j_to_e
                    + w_vqe * vq_e
                )

        if self.smpl_pose is not None:
            if x_smpl_pose is None:
                raise ValueError("smpl_pose modality is enabled but x_smpl_pose is None")
            # Intentionally disabled from optimization in the current training setup.
            stats["smpl_pose/enabled_but_unused"] = torch.tensor(1.0, device=total.device)

        stats["total_loss"] = total.detach()
        stats["vq/num_resets"] = self.vq.num_resets.detach().float()
        return total, stats

