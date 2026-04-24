from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# File path:
#   .../golf_third_party/musclesinaction/custom/stage2/train/train_stage2_pose2emg.py
_MIA_ROOT = Path(__file__).resolve().parents[3]
if str(_MIA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIA_ROOT))

from custom.models.frame_codebook import FrameCodebookModel  # noqa: E402
from custom.models.vq_ema import VQEMAConfig  # noqa: E402
from custom.models.frame_codebook import FrameCodebookConfig, ModalityConfig  # noqa: E402
from custom.models.temporal import TemporalConfig  # noqa: E402
from custom.utils.mia_filelist import build_mia_train_filelist  # noqa: E402
from custom.utils.path_utils import get_musclesinaction_repo_root  # noqa: E402
from custom.stage2.models.stage2_pose2emg import Stage2Pose2EMG, Stage2Pose2EMGConfig  # noqa: E402


# 8 通道 EMG 名称，与 eval 脚本一致，用于 log 中各通道损失
MUSCLE_NAMES: tuple = (
    "rightquad", "leftquad", "rightham", "leftham",
    "rightglutt", "leftglutt", "leftbicep", "rightbicep",
)


class _NullLogger:
    def info(self, *args, **kwargs):
        return

    def warning(self, *args, **kwargs):
        return

    def exception(self, *args, **kwargs):
        return


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")



def _to_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x


def _root_center_joints3d(joints3d: torch.Tensor, root_index: int) -> torch.Tensor:
    root = joints3d[:, :, root_index : root_index + 1, :]
    return joints3d - root


def _emg_standardizer_std(standardizer) -> torch.Tensor:
    """与 OnlineStandardizer 一致的安全 std，用于归一化/反归一化。"""
    var_safe = torch.clamp(standardizer.var, min=1e-4)
    return torch.sqrt(var_safe + 1e-6)


def _emg_standardizer_stats_bt8(standardizer, *, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = standardizer.mean.to(device)
    std = _emg_standardizer_std(standardizer).to(device)
    m = int(mean.numel())
    if m == 8:
        mean_bt8 = mean.view(1, 1, 8).expand(1, int(t), 8)
        std_bt8 = std.view(1, 1, 8).expand(1, int(t), 8)
        return mean_bt8, std_bt8
    if m == int(t) * 8:
        mean_bt8 = mean.view(1, int(t), 8)
        std_bt8 = std.view(1, int(t), 8)
        return mean_bt8, std_bt8
    # Unified-clip Stage1 standardizer: m == clip_len*8 (e.g. clip_len=5 -> 40).
    # Repeat stats across time to match current Stage2 sequence length T.
    if m % 8 == 0:
        clip_len = m // 8
        if clip_len > 0 and int(t) % int(clip_len) == 0:
            mean_clip = mean.view(1, int(clip_len), 8).repeat(1, int(t) // int(clip_len), 1)
            std_clip = std.view(1, int(clip_len), 8).repeat(1, int(t) // int(clip_len), 1)
            return mean_clip, std_clip
    raise ValueError(f"Unsupported EMG standardizer dim={m}; expected 8 or T*8={int(t)*8}")


@torch.no_grad()
def _eval_one_epoch(
    model: Stage2Pose2EMG,
    loader: DataLoader,
    device: torch.device,
    *,
    joints3d_root_center: bool,
    joints3d_root_index: int,
    emg_standardizer=None,
    distributed: bool = False,
) -> Tuple[float, float]:
    """返回 (val_smooth_l1, val_rmse)，均在 raw 空间，便于与 official_global_rmse 对照。"""
    model.eval()
    sum_smooth_l1 = 0.0
    sum_mse = 0.0
    total_samples = 0
    for batch in loader:
        joints3d = _to_tensor(batch["3dskeleton"]).to(device=device, dtype=torch.float32)  # (B,T,25,3)
        emg = _to_tensor(batch["emg_values"]).to(device=device, dtype=torch.float32)  # (B,8,T)
        b_sz = joints3d.shape[0]
        
        cond = _to_tensor(batch.get("condval", None))
        if cond is not None:
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.ndim == 1:
                cond = cond.unsqueeze(1)
        else:
            cond = None

        if joints3d_root_center:
            joints3d = _root_center_joints3d(joints3d, joints3d_root_index)
            
        gt = emg.permute(0, 2, 1).contiguous()  # (B,T,8)
        t = int(gt.shape[1])
        out = model(joints3d, cond=cond)
        pred = out["emg_pred"]
        if emg_standardizer is not None:
            mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=t, device=device)
            pred = pred * std_bt8 + mean_bt8
        sum_smooth_l1 += float(torch.nn.functional.smooth_l1_loss(pred, gt, reduction="sum").item())
        sum_mse += float(torch.nn.functional.mse_loss(pred, gt, reduction="sum").item())
        total_samples += b_sz * t * 8
    
    model.train()
    
    if distributed:
        t_metrics = torch.tensor([sum_smooth_l1, sum_mse, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(t_metrics, op=dist.ReduceOp.SUM)
        sum_smooth_l1 = float(t_metrics[0].item())
        sum_mse = float(t_metrics[1].item())
        total_samples = float(t_metrics[2].item())

    n = max(total_samples, 1)
    avg_smooth_l1 = sum_smooth_l1 / n
    avg_mse = sum_mse / n
    avg_rmse = math.sqrt(avg_mse)
    return float(avg_smooth_l1), float(avg_rmse)


def _build_stage1_from_ckpt(ckpt_path: Path, device: torch.device) -> FrameCodebookModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or "config" not in payload or "model_state" not in payload:
        raise RuntimeError(f"Stage1 checkpoint missing keys (config/model_state): {ckpt_path}")
    cfg = payload["config"]

    # Build stage1 model from embedded config
    model_cfg = cfg["model"]
    vq_kwargs = dict(model_cfg["vq"])
    if "beta" not in vq_kwargs and "commitment_weight" in vq_kwargs:
        vq_kwargs["beta"] = vq_kwargs.pop("commitment_weight")
    vq_cfg = VQEMAConfig(**vq_kwargs)
    mods = model_cfg["modalities"]

    def _mod(name: str, d: Dict[str, Any]) -> ModalityConfig:
        from custom.utils.online_standardize import OnlineStandardizeConfig  # local import

        std_cfg = OnlineStandardizeConfig(**(d.get("std", {}) or {}))
        temporal_cfg = TemporalConfig(**(d.get("temporal", {}) or {}))
        return ModalityConfig(
            name=name,
            in_dim=int(d["in_dim"]),
            hidden_dim=int(d["hidden_dim"]),
            token_count=int(d["token_count"]),
            code_dim=int(d["code_dim"]),
            encoder_type=str(d.get("encoder_type", "mixer")),
            decoder_type=str(d.get("decoder_type", "mixer")),
            recon_weight=float(d.get("recon_weight", 1.0)),
            online_std=bool(d.get("online_std", True)),
            std_cfg=std_cfg,
            temporal=temporal_cfg,
            mixer=d.get("mixer"),
            frame_mixer=d.get("frame_mixer"),
        )

    stage1_cfg = FrameCodebookConfig(
        vq=vq_cfg,
        joints3d=_mod("joints3d", mods["joints3d"]),
        smpl_pose=None,
        emg=_mod("emg", mods["emg"]),
        encoder_decoder_only=bool(model_cfg.get("encoder_decoder_only", False)),
    )
    m = FrameCodebookModel(stage1_cfg).to(device)
    m.load_state_dict(payload["model_state"], strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _dist_barrier(device: torch.device) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[int(device.index)] if device.index is not None else None)
        return
    dist.barrier()


def _run_training(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    distributed: bool,
) -> None:
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    _set_seed(int(seed) + int(rank))

    mia_root = get_musclesinaction_repo_root()
    if str(mia_root) not in sys.path:
        sys.path.insert(0, str(mia_root))
    os.chdir(str(mia_root))

    runtime = cfg["runtime"]
    data_cfg = cfg["data"]
    opt_cfg = cfg["optimizer"]

    if args.train_filelist is not None:
        cfg.setdefault("data", {})["train_filelist"] = args.train_filelist
    if args.val_filelist is not None:
        cfg.setdefault("data", {})["val_filelist"] = args.val_filelist

    train_filelist = Path(str(data_cfg["train_filelist"]))
    if not train_filelist.exists() or train_filelist.stat().st_size == 0:
        res = build_mia_train_filelist(
            mia_repo_root=mia_root,
            split=str(data_cfg.get("split", "train")),
            out_txt=train_filelist,
            max_samples=data_cfg.get("max_samples", None),
            require_files=data_cfg.get("require_files", ["emgvalues.npy", "joints3d.npy"]),
        )
        if rank == 0:
            print(f"[FileList] wrote {res.num_samples} samples -> {res.filelist_path}")

    from musclesinaction.dataloader.data import MyMuscleDataset, _seed_worker  # type: ignore

    dset = MyMuscleDataset(
        str(train_filelist),
        _NullLogger(),
        str(data_cfg.get("split", "train")),
        percent=float(data_cfg.get("percent", 1.0)),
        step=int(data_cfg.get("step", 30)),
        std=str(data_cfg.get("std", "False")),
        cond=str(data_cfg.get("cond", "True")),
        transform=None,
    )

    train_sampler = None
    shuffle = True
    if distributed:
        train_sampler = DistributedSampler(
            dset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            drop_last=True,
        )
        shuffle = False

    loader = DataLoader(
        dset,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=int(runtime.get("num_workers", 4)),
        drop_last=True,
        worker_init_fn=_seed_worker,
        pin_memory=(device.type == "cuda"),
    )

    is_main = int(rank) == 0

    val_loader = None
    val_filelist = Path(str(data_cfg.get("val_filelist", ""))) if data_cfg.get("val_filelist", None) else None
    if val_filelist is None or str(val_filelist).strip() == "":
        val_filelist = mia_root / "custom" / "tools" / "datasetsplits" / "miaofficial_val_eval.txt"
    val_filelist = Path(str(val_filelist))
    if is_main and (not val_filelist.exists() or val_filelist.stat().st_size == 0):
        res = build_mia_train_filelist(
            mia_repo_root=mia_root,
            split="val",
            out_txt=val_filelist,
            max_samples=data_cfg.get("max_samples", None),
            require_files=data_cfg.get("require_files", ["emgvalues.npy", "joints3d.npy"]),
        )
        print(f"[FileList] wrote {res.num_samples} samples -> {res.filelist_path}")
    if distributed:
        dist.barrier() # wait for rank 0 to build val filelist if needed

    if val_filelist.exists() and val_filelist.stat().st_size > 0:
        val_dset = MyMuscleDataset(
            str(val_filelist),
            _NullLogger(),
            "val",
            percent=1.0,
            step=int(data_cfg.get("step", 30)),
            std=str(data_cfg.get("std", "False")),
            cond=str(data_cfg.get("cond", "True")),
            transform=None,
        )
        val_sampler = None
        if distributed:
            val_sampler = DistributedSampler(
                val_dset,
                num_replicas=int(world_size),
                rank=int(rank),
                shuffle=False,
                drop_last=False,
            )
        val_loader = DataLoader(
            val_dset,
            batch_size=int(data_cfg.get("batch_size", 8)),
            shuffle=False,
            sampler=val_sampler,
            num_workers=int(runtime.get("num_workers", 4)),
            drop_last=False,
            worker_init_fn=_seed_worker,
            pin_memory=(device.type == "cuda"),
        )

    stage1_ckpt = Path(str(cfg["stage1"]["checkpoint"])).expanduser().resolve()
    stage1 = _build_stage1_from_ckpt(stage1_ckpt, device=device)

    model_cfg = cfg["model"]
    from custom.stage2.models.dcsa import DCSAConfig
    from custom.stage2.models.dstformer import DSTFormerConfig
    from custom.stage2.models.fusion import ResidualAddConfig
    from custom.stage2.models.temporal_backbone import TCNBackboneConfig

    fusion_type = str(model_cfg.get("fusion_type", "dcsa")).strip().lower()
    temporal_type = str(model_cfg.get("temporal_type", "dstformer")).strip().lower()
    dcsa_cfg = DCSAConfig(**(model_cfg.get("dcsa", {}) or {}))
    dst_cfg = DSTFormerConfig(**(model_cfg.get("dst", {}) or {}))
    from custom.stage2.models.dstformer_v2 import DSTFormerV2Config
    dst_v2_cfg = DSTFormerV2Config(**(model_cfg.get("dst_v2", {}) or {}))
    from custom.stage2.models.dstformer_v3_moe import DSTFormerV3MoEConfig
    dst_v3_moe_cfg = DSTFormerV3MoEConfig(**(model_cfg.get("dst_v3_moe", {}) or {}))
    dim = int(model_cfg.get("dim", 256))
    residual_add_cfg = None
    if fusion_type in ("residual_add", "residual", "add"):
        _r = dict(model_cfg.get("fusion_residual_add", {}) or {})
        _r["dim"] = dim
        residual_add_cfg = ResidualAddConfig(**_r)
    tcn_cfg = None
    if temporal_type == "tcn":
        _t = dict(model_cfg.get("tcn", {}) or {})
        _t["dim"] = dim
        tcn_cfg = TCNBackboneConfig(**_t)

    stage2_cfg = Stage2Pose2EMGConfig(
        token_count=int(model_cfg.get("token_count", 63)),
        dim=int(model_cfg.get("dim", 256)),
        cont_encoder_type=str(model_cfg.get("cont_encoder_type", "mixer")).strip().lower(),
        cont_hidden_dim=int(model_cfg.get("cont_hidden_dim", 1024)),
        cont_joint_hidden_dim=int(model_cfg.get("cont_joint_hidden_dim", 128)),
        fusion_type=fusion_type,
        dcsa=dcsa_cfg,
        fusion_residual_add=residual_add_cfg,
        temporal_type=temporal_type,
        dst=dst_cfg,
        dst_v2=dst_v2_cfg,
        dst_v3_moe=dst_v3_moe_cfg,
        tcn=tcn_cfg,
        emg_head_type=str(model_cfg.get("emg_head_type", "mixer")).strip().lower(),
        emg_hidden=int(model_cfg.get("emg_hidden", 256)),
        emg_mixer_hidden_dim=int(model_cfg.get("emg_mixer_hidden_dim", 256)),
        emg_mixer_num_layers=int(model_cfg.get("emg_mixer_num_layers", 4)),
        emg_pred_mode=str(model_cfg.get("emg_pred_mode", "full")).strip().lower(),
        max_seq_len=int(model_cfg.get("max_seq_len", 256)),
        use_cond=bool(model_cfg.get("use_cond", False)),
    )

    model = Stage2Pose2EMG(stage2_cfg, stage1=stage1).to(device)
    if distributed:
        model = DDP(model, device_ids=[int(device.index)], output_device=int(device.index), broadcast_buffers=False)

    base_lr = float(opt_cfg.get("lr", 2e-4))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    joints3d_root_center = bool(data_cfg.get("joints3d_root_center", True))
    joints3d_root_index = int(data_cfg.get("joints3d_root_index", 8))
    max_steps = int(runtime.get("max_steps", 3000))
    lr_min = float(opt_cfg.get("lr_min", 1e-6))
    lr_schedule = str(opt_cfg.get("lr_schedule", "none")).strip().lower()
    if lr_schedule == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max_steps, eta_min=lr_min
        )
    else:
        scheduler = None
    log_every = int(runtime.get("log_every", 50))
    save_every = int(runtime.get("save_every", 500))
    eval_every = int(runtime.get("eval_every", 500))
    grad_clip_norm = float(runtime.get("grad_clip_norm", 1.0))

    model_ref = model.module if isinstance(model, DDP) else model
    pred_mode = str(model_cfg.get("emg_pred_mode", "full")).strip().lower()
    emg_normalize_target = bool(data_cfg.get("emg_normalize_target", True))
    stage1_emg = getattr(model_ref.stage1, "emg", None)
    emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
    use_emg_norm = (
        pred_mode == "full"
        and emg_normalize_target
        and emg_standardizer is not None
    )
    if is_main and use_emg_norm:
        print("[Stage2Train] EMG target 使用 Stage1 Standardizer 归一化（监督在 0/1 空间，指标在 raw 空间）")

    ckpt_dir_cfg = str(cfg.get("checkpoints", {}).get("dir", "custom/stage2/checkpoints/stage2_pose2emg"))
    ckpt_dir_path = Path(ckpt_dir_cfg).expanduser()
    ckpt_dir = (ckpt_dir_path if ckpt_dir_path.is_absolute() else (mia_root / ckpt_dir_path)).resolve()
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    best_val_rmse = float("inf")

    if is_main:
        if args.ckpt_dir is not None:
            log_path = ckpt_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_dir = (mia_root / "custom" / "stage2" / "logs").resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _log_file = open(log_path, "w", encoding="utf-8")
        _orig_stdout = sys.stdout

        class _Tee:
            def __init__(self, stdout, f):
                self._stdout = stdout
                self._file = f

            def write(self, s):
                self._stdout.write(s)
                self._file.write(s)
                self._file.flush()

            def flush(self):
                self._stdout.flush()
                self._file.flush()

        if args.ckpt_dir is not None:
            temp_data_dir = (ckpt_dir / "temp_data").resolve()
        else:
            # Change temp_data_dir to /data/litengmo/tmp/stage2 to prevent out-of-space issues
            temp_data_dir = Path("/data/litengmo/tmp/stage2").resolve()
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        train_data_paths_file = temp_data_dir / "train_data_paths.txt"
        train_data_paths_handle = open(train_data_paths_file, "a", encoding="utf-8")
        seen_data_paths = set()

        sys.stdout = _Tee(_orig_stdout, _log_file)
    else:
        _orig_stdout = sys.stdout
        _log_file = None
        temp_data_dir = None
        train_data_paths_file = None
        train_data_paths_handle = None
        seen_data_paths = None
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    try:
        if is_main:
            print("[Stage2Train] ========== Config ==========")
            print(yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False))
            print("[Stage2Train] fusion_type=%s temporal_type=%s | ckpt_dir=%s | max_steps=%d" % (fusion_type, temporal_type, ckpt_dir, max_steps))
            print("[Stage2Train] ==============================")
            print(f"[Stage2Train] log file -> {log_path}")

        if args.init_stage2_checkpoint is not None:
            init_path = Path(str(args.init_stage2_checkpoint)).expanduser()
            init_path = init_path if init_path.is_absolute() else (mia_root / init_path).resolve()
            payload = torch.load(init_path, map_location="cpu")
            if not isinstance(payload, dict) or "model_state" not in payload:
                raise RuntimeError(f"Invalid stage2 init checkpoint: {init_path}")
            (model_ref).load_state_dict(payload["model_state"], strict=True)
            if is_main:
                print(f"[Stage2Train] warm-start from -> {init_path}")
            if distributed:
                dist.barrier()

        model.train()
        start_time = time.time()
        it = iter(loader)
        step = 0
        if bool(args.resume) and last_path.exists():
            payload = torch.load(last_path, map_location="cpu")
            if isinstance(payload, dict) and "model_state" in payload and "optim_state" in payload:
                (model_ref).load_state_dict(payload["model_state"], strict=True)
                optim.load_state_dict(payload["optim_state"])
                if scheduler is not None and isinstance(payload.get("scheduler_state", None), dict):
                    scheduler.load_state_dict(payload["scheduler_state"])
                step = int(payload.get("step", 0))
                best_val_rmse = float(payload.get("best_val_rmse", best_val_rmse))
                if is_main:
                    print(f"[Stage2Train] resume from -> {last_path} (step={step}, best_val_rmse={best_val_rmse})")
            else:
                raise RuntimeError(f"Invalid last checkpoint payload: {last_path}")
            if distributed:
                dist.barrier()

        while step < max_steps:
            if distributed and train_sampler is not None and step % max(len(loader), 1) == 0:
                train_sampler.set_epoch(int(step // max(len(loader), 1)))
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            if is_main and train_data_paths_handle is not None and seen_data_paths is not None:
                paths_this_batch = batch.get("filepath")
                if paths_this_batch is not None:
                    if not isinstance(paths_this_batch, (list, tuple)):
                        paths_this_batch = [paths_this_batch]
                    for p in paths_this_batch:
                        path_str = str(p).strip()
                        if path_str and path_str not in seen_data_paths:
                            seen_data_paths.add(path_str)
                        if path_str:
                            train_data_paths_handle.write(path_str + "\n")
                    train_data_paths_handle.flush()

            joints3d = _to_tensor(batch["3dskeleton"]).to(device=device, dtype=torch.float32)
            emg = _to_tensor(batch["emg_values"]).to(device=device, dtype=torch.float32)

            cond = _to_tensor(batch.get("condval", None))
            if cond is not None:
                cond = cond.to(device=device, dtype=torch.float32)
                if cond.ndim == 1:
                    cond = cond.unsqueeze(1)
            else:
                cond = None

            if joints3d_root_center:
                joints3d = _root_center_joints3d(joints3d, joints3d_root_index)
            gt_raw = emg.permute(0, 2, 1).contiguous()

            out = model(joints3d, cond=cond)
            pred = out["emg_pred"]
            if use_emg_norm:
                mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(gt_raw.shape[1]), device=device)
                gt_norm = (gt_raw - mean_bt8) / std_bt8
                loss = torch.nn.functional.smooth_l1_loss(pred, gt_norm)
                pred_raw = (pred * std_bt8 + mean_bt8).detach()
            else:
                loss = torch.nn.functional.smooth_l1_loss(pred, gt_raw)
                pred_raw = pred.detach()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            step += 1
            if is_main and (step % log_every == 0 or step == 1):
                dt = max(time.time() - start_time, 1e-6)
                fps = (step * int(data_cfg.get("batch_size", 8)) * int(data_cfg.get("step", 30))) / dt
                current_loss = float(loss.item())
                current_loss_raw = float(torch.nn.functional.smooth_l1_loss(pred_raw, gt_raw).item())
                with torch.no_grad():
                    loss_mse = float(torch.nn.functional.mse_loss(pred_raw, gt_raw).item())
                    loss_mae = float(torch.nn.functional.l1_loss(pred_raw, gt_raw).item())
                    per_ch_smooth_l1 = []
                    for ch in range(8):
                        per_ch_smooth_l1.append(
                            float(torch.nn.functional.smooth_l1_loss(pred_raw[..., ch], gt_raw[..., ch]).item())
                        )
                msg = {
                    "step": step,
                    "loss_smooth_l1": current_loss_raw,
                    "loss_mse": loss_mse,
                    "loss_mae": loss_mae,
                    "lr": optim.param_groups[0]["lr"],
                    "fps_frames_per_sec": float(fps),
                }
                if use_emg_norm:
                    msg["loss_smooth_l1_norm"] = current_loss
                for ch, name in enumerate(MUSCLE_NAMES):
                    msg[f"ch_{name}_smooth_l1"] = per_ch_smooth_l1[ch]
                print("[Stage2Train]", msg)

            do_eval = (eval_every > 0) and (step % eval_every == 0 or step == max_steps)
            if do_eval and val_loader is not None:
                val_smooth_l1, val_rmse = _eval_one_epoch(
                    model_ref,
                    val_loader,
                    device,
                    joints3d_root_center=joints3d_root_center,
                    joints3d_root_index=joints3d_root_index,
                    emg_standardizer=emg_standardizer if use_emg_norm else None,
                    distributed=distributed,
                )
                if is_main:
                    if val_rmse < best_val_rmse:
                        best_val_rmse = float(val_rmse)
                        payload = {
                            "step": int(step),
                            "best_val_rmse": float(best_val_rmse),
                            "config": cfg,
                            "stage1_checkpoint": str(stage1_ckpt),
                            "model_state": model_ref.state_dict(),
                            "optim_state": optim.state_dict(),
                        }
                        if scheduler is not None:
                            payload["scheduler_state"] = scheduler.state_dict()
                        torch.save(payload, best_path)
                        print(f"[Checkpoint] saved best (val_rmse={best_val_rmse:.6f}) -> {best_path}")
                    print("[Stage2Val]", {"step": step, "val_smooth_l1": val_smooth_l1, "val_rmse": val_rmse, "best_val_rmse": float(best_val_rmse)})

            do_save = (save_every > 0) and (step % save_every == 0 or step == max_steps)
            if is_main and do_save:
                payload = {
                    "step": int(step),
                    "best_val_rmse": float(best_val_rmse),
                    "config": cfg,
                    "stage1_checkpoint": str(stage1_ckpt),
                    "model_state": model_ref.state_dict(),
                    "optim_state": optim.state_dict(),
                }
                if scheduler is not None:
                    payload["scheduler_state"] = scheduler.state_dict()
                torch.save(payload, last_path)
                print(f"[Checkpoint] saved last -> {last_path}")
            if distributed and do_save:
                _dist_barrier(device)

    finally:
        sys.stdout = _orig_stdout
        if is_main and _log_file is not None:
            _log_file.close()
        if is_main and train_data_paths_handle is not None and temp_data_dir is not None and train_data_paths_file is not None and seen_data_paths is not None:
            train_data_paths_handle.close()
            summary_path = temp_data_dir / "train_data_paths_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"total_unique_paths: {len(seen_data_paths)}\n")
                with open(train_data_paths_file, encoding="utf-8") as rf:
                    n_lines = sum(1 for _ in rf)
                f.write(f"total_lines_in_train_data_paths_txt: {n_lines}\n")
            print(f"[TempData] 参与训练的不同数据数: {len(seen_data_paths)}，路径已追加到 {train_data_paths_file}，统计见 {summary_path}")


def _ddp_spawn(
    local_rank: int,
    world_size: int,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    master_addr: str,
    master_port: int,
    backend: str,
) -> None:
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(int(local_rank))
    dist.init_process_group(
        backend=str(backend),
        init_method=f"tcp://{master_addr}:{int(master_port)}",
        rank=int(local_rank),
        world_size=int(world_size),
    )
    try:
        _run_training(
            cfg,
            args,
            device=torch.device(f"cuda:{int(local_rank)}"),
            rank=int(local_rank),
            world_size=int(world_size),
            distributed=True,
        )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="custom/stage2/configs/stage2_pose2emg.yaml",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Override stage1 checkpoint path (otherwise use config stage1.checkpoint).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cuda:0, cuda:4). Default from config runtime.device.",
    )
    parser.add_argument(
        "--train_filelist",
        type=str,
        default=None,
        help="Override data.train_filelist (supports absolute or relative to mia_root).",
    )
    parser.add_argument(
        "--val_filelist",
        type=str,
        default=None,
        help="Override data.val_filelist (supports absolute or relative to mia_root).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Override checkpoints.dir (supports absolute or relative to mia_root).",
    )
    parser.add_argument(
        "--init_stage2_checkpoint",
        type=str,
        default=None,
        help="Optional: load model_state from a stage2 checkpoint before training (warm-start).",
    )
    parser.add_argument(
        "--resume",
        type=_str2bool,
        default=False,
        help="If true and last.pt exists under ckpt_dir, resume model/optim/scheduler/step from it.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Override runtime.max_steps.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer.lr.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override data.batch_size.")
    parser.add_argument("--save_every", type=int, default=None, help="Override runtime.save_every.")
    parser.add_argument("--eval_every", type=int, default=None, help="Override runtime.eval_every.")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config).expanduser().resolve())
    if args.stage1_checkpoint is not None:
        cfg.setdefault("stage1", {})["checkpoint"] = args.stage1_checkpoint
    if args.train_filelist is not None:
        cfg.setdefault("data", {})["train_filelist"] = args.train_filelist
    if args.val_filelist is not None:
        cfg.setdefault("data", {})["val_filelist"] = args.val_filelist
    if args.ckpt_dir is not None:
        cfg.setdefault("checkpoints", {})["dir"] = args.ckpt_dir
    if args.max_steps is not None:
        cfg.setdefault("runtime", {})["max_steps"] = int(args.max_steps)
    if args.lr is not None:
        cfg.setdefault("optimizer", {})["lr"] = float(args.lr)
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = int(args.batch_size)
    if args.save_every is not None:
        cfg.setdefault("runtime", {})["save_every"] = int(args.save_every)
    if args.eval_every is not None:
        cfg.setdefault("runtime", {})["eval_every"] = int(args.eval_every)
    runtime = cfg["runtime"]

    dist_cfg = (runtime.get("distributed", {}) or {}) if isinstance(runtime, dict) else {}
    dist_enabled = bool(dist_cfg.get("enabled", False))
    backend = str(dist_cfg.get("backend", "nccl"))
    gpus = dist_cfg.get("gpus", None)

    if dist_enabled and torch.cuda.is_available():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(int(local_rank))
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, init_method="env://")
            try:
                _run_training(
                    cfg,
                    args,
                    device=torch.device(f"cuda:{int(local_rank)}"),
                    rank=int(rank),
                    world_size=int(world_size),
                    distributed=True,
                )
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
            return

        if gpus is None:
            gpus = list(range(int(torch.cuda.device_count())))
        gpus = [int(x) for x in gpus]
        if len(gpus) <= 1:
            _run_training(
                cfg,
                args,
                device=torch.device(str(runtime.get("device", "cuda"))),
                rank=0,
                world_size=1,
                distributed=False,
            )
            return
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpus)
        world_size = len(gpus)
        master_addr = str(dist_cfg.get("master_addr", "127.0.0.1"))
        master_port = int(dist_cfg.get("master_port", _free_port()))
        mp.spawn(
            _ddp_spawn,
            args=(int(world_size), cfg, args, master_addr, int(master_port), backend),
            nprocs=int(world_size),
            join=True,
        )
        return

    if args.device is not None:
        device = torch.device(args.device)
    else:
        dev_str = str(runtime.get("device", "cuda"))
        device = torch.device("cuda" if (dev_str == "cuda" and torch.cuda.is_available()) else dev_str)

    _run_training(
        cfg,
        args,
        device=device,
        rank=0,
        world_size=1,
        distributed=False,
    )


if __name__ == "__main__":
    main()
