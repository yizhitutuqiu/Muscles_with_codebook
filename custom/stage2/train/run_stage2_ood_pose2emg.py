from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _mia_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "custom").exists() and (parent / "musclesinaction").exists():
            return parent
    return p.parents[3]


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _extract_cfg(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        cfg = payload.get("config", None) or payload.get("cfg", None)
        if isinstance(cfg, dict):
            return cfg
        if isinstance(cfg, str):
            return yaml.safe_load(cfg) or {}
    raise ValueError("Cannot find config dict in checkpoint payload.")


def _ensure_single_gpu_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime = cfg.setdefault("runtime", {}) or {}
    if not isinstance(runtime, dict):
        runtime = {}
        cfg["runtime"] = runtime
    dist = runtime.get("distributed", {}) or {}
    if not isinstance(dist, dict):
        dist = {}
    dist["enabled"] = False
    dist.pop("gpus", None)
    runtime["distributed"] = dist
    runtime["device"] = "cuda:0"
    return cfg


def _latest_run_dir(parent: Path, prefix: str) -> Optional[Path]:
    if not parent.exists():
        return None
    cands = sorted([p for p in parent.glob(f"{prefix}*") if p.is_dir()])
    return cands[-1] if cands else None


def _targets_from_split_dir(split_dir: Path, only: Optional[List[str]]) -> List[str]:
    vals = sorted([p for p in split_dir.glob("val_*.txt") if p.is_file()])
    targets: List[str] = []
    for p in vals:
        stem = p.stem
        if not stem.startswith("val_"):
            continue
        t = stem[len("val_") :].strip()
        if t:
            targets.append(t)
    if only:
        s = set(only)
        targets = [t for t in targets if t in s]
    return targets


def _parse_stage2_average(summary_processed_csv: Path) -> float:
    if not summary_processed_csv.exists():
        raise FileNotFoundError(f"summary_processed.csv not found: {summary_processed_csv}")
    with open(summary_processed_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if str(r.get("exercise", "")).strip().lower() == "average":
            v = str(r.get("stage2", "")).strip()
            if v:
                return float(v)
            break
    summary_csv = summary_processed_csv.parent / "summary.csv"
    if summary_csv.exists():
        with open(summary_csv, "r", encoding="utf-8") as f:
            rows2 = list(csv.DictReader(f))
        vals = []
        for r in rows2:
            if str(r.get("method", "")).strip() != "stage2":
                continue
            s = str(r.get("official_global_rmse", "")).strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except Exception:
                continue
        if vals:
            return float(sum(vals) / len(vals))
    raise RuntimeError(f"Cannot parse stage2 average from: {summary_processed_csv}")


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_summary_md(path: Path, rows: List[Dict[str, Any]], title: str, base_dir: Path, split_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append("### 说明")
    lines.append("")
    lines.append(f"- 基础实验配置来源（best.pt）：`{base_dir}`")
    lines.append(f"- OOD split 目录：`{split_dir}`")
    lines.append("- 训练：每个 target 使用其对应的 train_{target}.txt / val_{target}.txt，串行训练 15 组。")
    lines.append("- 评测：每个 target 训练完成后，用对应的 train/val filelist 做一次评测，并写入该子目录 eval_results。")
    lines.append("")
    lines.append("### 均值汇总（只写 Average）")
    lines.append("")
    lines.append("| target | stage2_average | ckpt_dir | summary_processed |")
    lines.append("|---|---:|---|---|")
    for r in rows:
        lines.append(
            f"| {r['target']} | {float(r['stage2_average']):.4f} | {r['ckpt_dir']} | {r['summary_processed']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_eval_yaml(
    *,
    out_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    split_dir: Path,
    target: str,
    stage2_best: Path,
    stage1_checkpoint: Optional[str],
) -> Path:
    cfg: Dict[str, Any] = {
        "runtime": {"device": device, "num_workers": int(num_workers), "batch_size": int(batch_size)},
        "data": {"step": 30, "std": False, "percent": 1.0, "joints3d_root_center": True},
        "protocols": [
            {
                "name": "ood_exercises",
                "enabled": True,
                "split": "val",
                "ablation_dir": str(split_dir),
                "train_filelist": f"train_{target}.txt",
                "val_filelists": [f"val_{target}.txt"],
            }
        ],
        "output": {"out_dir": str(out_dir)},
        "methods": {
            "stage2": {"enabled": True, "checkpoint": str(stage2_best)},
            "official_cond": {"enabled": False},
            "official_nocond": {"enabled": False},
            "retrieval": {"enabled": True, "max_train_samples": None},
        },
    }
    if stage1_checkpoint:
        cfg["methods"]["stage2"]["stage1_checkpoint"] = stage1_checkpoint
    tmp = out_dir.parent / "temp_eval.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp


def _run_stage2_train_with_step_tqdm(
    *,
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    target: str,
    max_steps: int,
) -> None:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=int(max_steps), desc=f"train[{target}]", leave=False, dynamic_ncols=True)

    step_re = re.compile(r"'step':\s*(\d+)")
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    last_step = 0
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = step_re.search(line)
        if m:
            try:
                s = int(m.group(1))
                if pbar is not None and s > last_step:
                    pbar.update(s - last_step)
                last_step = max(last_step, s)
            except Exception:
                pass
    rc = p.wait()
    if pbar is not None:
        pbar.close()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _extract_stage1_ckpt_from_stage2_best(stage2_best: Path) -> Optional[str]:
    payload = _torch_load(stage2_best)
    if isinstance(payload, dict):
        v = payload.get("stage1_checkpoint", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/batch_stage2_h8_lightweight_0511/exp_h8_baseline_8exp",
    )
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument(
        "--split_dir",
        type=str,
        default="/data/litengmo/HSMR/golf_third_party/musclesinaction/musclesinaction/ablation/generalizationexercises",
    )
    parser.add_argument("--only", type=str, default="", help="Comma-separated targets (e.g. Running,Squat).")
    parser.add_argument(
        "--out_root",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/ood",
    )
    parser.add_argument("--max_steps", type=int, default=16000)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, skip training and only evaluate existing per-target checkpoints under an existing run dir.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Existing run directory to use in eval_only mode (e.g. .../pose2emg_ood_h8_8exp_YYYYmmdd_HHMMSS). If empty, auto-pick latest under out_root.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default="false")
    args = parser.parse_args()

    mia_root = _mia_root()
    base_dir = Path(args.base_dir).expanduser().resolve()
    base_best = base_dir / "best.pt"
    if not base_best.exists():
        raise FileNotFoundError(f"best.pt not found: {base_best}")

    split_dir = Path(args.split_dir).expanduser().resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"split_dir not found: {split_dir}")

    only = [x.strip() for x in str(args.only).split(",") if x.strip()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root_parent = Path(args.out_root).expanduser().resolve()
    out_root_parent.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        if str(args.run_dir).strip():
            out_root = Path(str(args.run_dir)).expanduser().resolve()
        else:
            out_root = _latest_run_dir(out_root_parent, "pose2emg_ood_h8_8exp_") or out_root_parent
        if not (out_root / "checkpoints").exists():
            raise FileNotFoundError(f"Existing run dir does not contain checkpoints/: {out_root}")
        cfg_path = out_root / "base_stage2_config.yaml"
    else:
        targets = _targets_from_split_dir(split_dir, only if only else None)
        if not targets:
            raise RuntimeError("No targets found to run.")
        payload = _torch_load(base_best)
        base_cfg = _extract_cfg(payload)
        base_cfg = _ensure_single_gpu_cfg(base_cfg)
        base_cfg.setdefault("runtime", {})["max_steps"] = int(args.max_steps)
        out_root = out_root_parent / f"pose2emg_ood_h8_8exp_{stamp}"
        out_root.mkdir(parents=True, exist_ok=True)
        cfg_path = out_root / "base_stage2_config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    env["PYTHONPATH"] = ":".join([str(mia_root), env.get("PYTHONPATH", "")]).strip(":")
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_SILENT"] = "true"
    env["MPLBACKEND"] = "Agg"

    train_script = mia_root / "custom/stage2/train/train_stage2_pose2emg.py"
    eval_script = mia_root / "custom/tools/Mia_style_eval.py"
    if not train_script.exists():
        raise FileNotFoundError(f"train script not found: {train_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    rows: List[Dict[str, Any]] = []
    if args.eval_only:
        ckpt_root = (out_root / "checkpoints").resolve()
        targets = sorted([p.name for p in ckpt_root.iterdir() if p.is_dir() and (p / "best.pt").exists()])
        if only:
            s = set(only)
            targets = [t for t in targets if t in s]
        if not targets:
            raise RuntimeError(f"No existing targets with best.pt found under: {ckpt_root}")
    it = targets
    if tqdm is not None:
        it = tqdm(targets, desc="OOD cases (pose2emg)", dynamic_ncols=True)
    for target in it:
        train_file = split_dir / f"train_{target}.txt"
        val_file = split_dir / f"val_{target}.txt"
        if not train_file.exists():
            raise FileNotFoundError(f"Missing train file: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Missing val file: {val_file}")

        ckpt_dir = (out_root / "checkpoints" / target).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        stage2_best = ckpt_dir / "best.pt"
        if not args.eval_only:
            cmd = [
                sys.executable,
                str(train_script),
                "--config",
                str(cfg_path),
                "--device",
                "cuda:0",
                "--train_filelist",
                str(train_file),
                "--val_filelist",
                str(val_file),
                "--ckpt_dir",
                str(ckpt_dir),
                "--resume",
                str(args.resume),
                "--max_steps",
                str(int(args.max_steps)),
            ]
            _run_stage2_train_with_step_tqdm(
                cmd=cmd,
                env=env,
                cwd=mia_root,
                target=str(target),
                max_steps=int(args.max_steps),
            )
        if not stage2_best.exists():
            raise FileNotFoundError(f"best.pt not found: {stage2_best}")

        eval_out_dir = ckpt_dir / "eval_results"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        stage1_checkpoint = _extract_stage1_ckpt_from_stage2_best(stage2_best)
        eval_yaml = _make_eval_yaml(
            out_dir=eval_out_dir,
            device="cuda:0",
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.eval_num_workers),
            split_dir=split_dir,
            target=target,
            stage2_best=stage2_best,
            stage1_checkpoint=stage1_checkpoint,
        )
        try:
            subprocess.run([sys.executable, str(eval_script), "--config", str(eval_yaml)], check=True, env=env, cwd=str(mia_root))
        finally:
            if eval_yaml.exists():
                eval_yaml.unlink()

        summary_processed = eval_out_dir / "summary_processed.csv"
        avg = _parse_stage2_average(summary_processed)
        rows.append(
            {
                "target": target,
                "stage2_average": f"{avg:.6f}",
                "ckpt_dir": str(ckpt_dir),
                "summary_processed": str(summary_processed),
            }
        )

    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(results_dir / "summary_ood_pose2emg.csv", rows)
    _write_summary_md(
        results_dir / "summary_ood_pose2emg.md",
        rows,
        title="Stage2 OOD Exercises (Pose2EMG) Summary",
        base_dir=base_dir,
        split_dir=split_dir,
    )
    print("[OK] out_root =", out_root)


if __name__ == "__main__":
    main()
