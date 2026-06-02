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


def _append_log(log_file: Path, msg: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def _parse_stage2_average(summary_processed_csv: Path) -> float:
    if summary_processed_csv.exists():
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


def _write_summary_md(path: Path, rows: List[Dict[str, Any]], title: str, base_cfg: Path, split_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append("### 说明")
    lines.append("")
    lines.append(f"- 基础实验配置来源（base_stage2_config.yaml）：`{base_cfg}`")
    lines.append(f"- person split 目录：`{split_dir}`")
    lines.append("- 训练：每个 subject 写一个 train_wo_SubjectK.txt，并串行训练。")
    lines.append("- 评测：每个 subject 训练完成后，用对应的 train/val filelist 做一次评测，并写入该子目录 eval_results。")
    lines.append("")
    lines.append("### 均值汇总（只写 Average）")
    lines.append("")
    lines.append("| subject | stage2_average | ckpt_dir | summary_processed |")
    lines.append("|---|---:|---|---|")
    for r in rows:
        lines.append(
            f"| {r['subject']} | {float(r['stage2_average']):.6f} | {r['ckpt_dir']} | {r['summary_processed']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_filelist(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def _write_filelist(p: Path, lines: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_id_val_file(*, mia_root: Path, cfg_path: Path, override: str) -> Path:
    if str(override).strip():
        p = Path(str(override)).expanduser()
        if not p.is_absolute():
            p = mia_root / p
        return p.resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    v = (cfg.get("data", {}) or {}).get("val_filelist", "") or ""
    if not str(v).strip():
        return (mia_root / "custom/tools/datasetsplits/miaofficial_val_eval.txt").resolve()
    p = Path(str(v)).expanduser()
    if not p.is_absolute():
        p = mia_root / p
    return p.resolve()


def _build_id_minus_heldout_val(*, id_val_file: Path, heldout_val_file: Path, out_file: Path) -> Tuple[int, int]:
    id_lines = _read_filelist(id_val_file)
    heldout_lines = _read_filelist(heldout_val_file)
    heldout_set = set(heldout_lines)
    kept = [ln for ln in id_lines if ln not in heldout_set]
    if not kept:
        raise RuntimeError(f"id_minus_heldout val is empty. id_val={id_val_file} heldout_val={heldout_val_file}")
    _write_filelist(out_file, kept)
    removed = len(id_lines) - len(kept)
    return len(id_lines), removed


def _make_eval_yaml(
    *,
    out_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    ablation_dir: Path,
    train_filelist: str,
    val_filelists: List[str],
    stage2_best: Path,
    stage1_checkpoint: Optional[str],
    stage1_checkpoint_aux: Optional[str],
) -> Path:
    cfg: Dict[str, Any] = {
        "runtime": {"device": device, "num_workers": int(num_workers), "batch_size": int(batch_size)},
        "data": {"step": 30, "std": False, "percent": 1.0, "joints3d_root_center": True},
        "protocols": [
            {
                "name": "ood_people",
                "enabled": True,
                "split": "val",
                "ablation_dir": str(ablation_dir),
                "train_filelist": train_filelist,
                "val_filelists": val_filelists,
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
    if stage1_checkpoint_aux:
        cfg["methods"]["stage2"]["stage1_checkpoint_aux"] = stage1_checkpoint_aux
    tmp = out_dir.parent / "temp_eval.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp


def _run_stage2_train_with_step_tqdm(
    *,
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    subject: str,
    max_steps: int,
) -> None:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=int(max_steps), desc=f"train[{subject}]", leave=False, dynamic_ncols=True)

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


def _extract_stage1_aux_ckpt_from_stage2_best(stage2_best: Path) -> Optional[str]:
    payload = _torch_load(stage2_best)
    if isinstance(payload, dict):
        v = payload.get("stage1_checkpoint_aux", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _subject_id_from_val_file(p: Path) -> str:
    m = re.match(r"valSubject(\d+)\.txt$", p.name)
    if not m:
        raise ValueError(f"Unexpected val file name: {p}")
    return f"Subject{int(m.group(1))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="",
        help="If set, load base config from <base_dir>/best.pt instead of --base_cfg yaml.",
    )
    parser.add_argument(
        "--base_cfg",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/lightweight_stage1_stage2/ood/pose2emg_ood_h8_8exp_20260521_165758/base_stage2_config.yaml",
    )
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument(
        "--split_dir",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/musclesinaction/ablation/generalization_ID_nocond_people",
    )
    parser.add_argument("--only", type=str, default="", help="Comma-separated subject ids to run (e.g. 0,1,2). Empty runs all.")
    parser.add_argument(
        "--out_root",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/lightweight_stage1_stage2/ood",
    )
    parser.add_argument("--max_steps", type=int, default=16000)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_num_workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default="false")
    parser.add_argument(
        "--best_val_mode",
        type=str,
        default="heldout",
        choices=["heldout", "id_minus_heldout"],
        help="Which validation set to use for selecting best.pt during training.",
    )
    parser.add_argument(
        "--id_val_filelist",
        type=str,
        default="",
        help="ID validation filelist used by best_val_mode=id_minus_heldout. If empty, inferred from base_stage2_config.yaml data.val_filelist.",
    )
    args = parser.parse_args()

    mia_root = _mia_root()
    base_cfg_path: Optional[Path] = None
    base_dir = Path(str(args.base_dir)).expanduser().resolve() if str(args.base_dir).strip() else None
    if base_dir is not None:
        base_best = base_dir / "best.pt"
        if not base_best.exists():
            raise FileNotFoundError(f"best.pt not found under base_dir: {base_best}")
    else:
        base_cfg_path = Path(args.base_cfg).expanduser().resolve()
        if not base_cfg_path.exists():
            raise FileNotFoundError(f"base_cfg not found: {base_cfg_path}")

    split_dir = Path(args.split_dir).expanduser().resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"split_dir not found: {split_dir}")
    base_train_list = split_dir / "train.txt"
    if not base_train_list.exists():
        raise FileNotFoundError(f"train.txt not found: {base_train_list}")

    only = [x.strip() for x in str(args.only).split(",") if x.strip()]
    only_set = {f"Subject{int(x)}" for x in only} if only else set()

    out_root_parent = Path(args.out_root).expanduser().resolve()
    out_root_parent.mkdir(parents=True, exist_ok=True)
    run_dir_arg = str(args.run_dir).strip()
    if args.eval_only:
        if run_dir_arg:
            out_root = Path(run_dir_arg).expanduser().resolve()
        else:
            out_root = _latest_run_dir(out_root_parent, "pose2emg_ood_person_") or out_root_parent
        if not (out_root / "checkpoints").exists():
            raise FileNotFoundError(f"Existing run dir does not contain checkpoints/: {out_root}")
        cfg_path = out_root / "base_stage2_config.yaml"
        splits_root = out_root / "splits"
    else:
        val_files = sorted([p for p in split_dir.glob("valSubject*.txt") if p.is_file()])
        subjects = [_subject_id_from_val_file(p) for p in val_files]
        if only_set:
            subjects = [s for s in subjects if s in only_set]
        if not subjects:
            raise RuntimeError("No subjects found to run.")
        if base_dir is not None:
            payload = _torch_load(base_best)
            base_cfg = _extract_cfg(payload)
        else:
            assert base_cfg_path is not None
            base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
            if not isinstance(base_cfg, dict):
                raise ValueError(f"Invalid base cfg: {base_cfg_path}")
        base_cfg = _ensure_single_gpu_cfg(base_cfg)
        base_cfg.setdefault("runtime", {})["max_steps"] = int(args.max_steps)
        if args.train_batch_size is not None:
            base_cfg.setdefault("data", {})["batch_size"] = int(args.train_batch_size)
        base_cfg.setdefault("model", {})
        if not base_cfg["model"].get("task"):
            base_cfg["model"]["task"] = "pose2emg"
        if run_dir_arg:
            out_root = Path(run_dir_arg).expanduser().resolve()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_root = out_root_parent / f"pose2emg_ood_person_{stamp}"
        out_root.mkdir(parents=True, exist_ok=True)
        cfg_path = out_root / "base_stage2_config.yaml"
        splits_root = out_root / "splits"
        splits_root.mkdir(parents=True, exist_ok=True)
        if not cfg_path.exists():
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(base_cfg, f, sort_keys=False)

        base_lines = [ln.strip() for ln in base_train_list.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        for s in subjects:
            train_out = splits_root / f"train_wo_{s}.txt"
            val_out = splits_root / f"val_{s}.txt"
            if not train_out.exists():
                keep = [ln for ln in base_lines if f"/{s}/" not in ln]
                if not keep:
                    raise RuntimeError(f"Filtered train list is empty for held-out {s}.")
                train_out.write_text("\n".join(keep) + "\n", encoding="utf-8")
            if not val_out.exists():
                val_src = split_dir / f"val{s}.txt"
                if not val_src.exists():
                    m = re.match(r"Subject(\d+)$", s)
                    if not m:
                        raise FileNotFoundError(f"val file missing for {s}")
                    val_src = split_dir / f"valSubject{int(m.group(1))}.txt"
                val_out.write_text(val_src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")

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

    subjects: List[str]
    if args.eval_only:
        ckpt_root = (out_root / "checkpoints").resolve()
        subjects = sorted([p.name for p in ckpt_root.iterdir() if p.is_dir() and (p / "best.pt").exists()])
        if only_set:
            subjects = [s for s in subjects if s in only_set]
        if not subjects:
            raise RuntimeError(f"No existing subjects with best.pt found under: {ckpt_root}")
    else:
        val_files = sorted([p for p in split_dir.glob("valSubject*.txt") if p.is_file()])
        subjects = [_subject_id_from_val_file(p) for p in val_files]
        if only_set:
            subjects = [s for s in subjects if s in only_set]

    rows: List[Dict[str, Any]] = []
    run_log = out_root / "runner.log"
    for s in subjects:
        try:
            ckpt_dir = (out_root / "checkpoints" / s).resolve()
            stage2_best = ckpt_dir / "best.pt"
            eval_out_dir = ckpt_dir / "eval_results"
            summary_processed = eval_out_dir / "summary_processed.csv"

            if stage2_best.exists() and (summary_processed.exists() or (eval_out_dir / "summary.csv").exists()):
                avg = _parse_stage2_average(summary_processed)
                rows.append(
                    {
                        "subject": s,
                        "stage2_average": f"{avg:.6f}",
                        "ckpt_dir": str(ckpt_dir),
                        "summary_processed": str(summary_processed),
                    }
                )
                _append_log(run_log, f"[SKIP] {s} already finished")
                continue

            ckpt_dir.mkdir(parents=True, exist_ok=True)

            train_file = splits_root / f"train_wo_{s}.txt"
            val_file = splits_root / f"val_{s}.txt"
            if not train_file.exists():
                raise FileNotFoundError(f"Missing train file: {train_file}")
            if not val_file.exists():
                raise FileNotFoundError(f"Missing val file: {val_file}")

            _append_log(run_log, f"[START] {s}")
            if not args.eval_only:
                if str(args.best_val_mode) == "id_minus_heldout":
                    id_val_file = _resolve_id_val_file(mia_root=mia_root, cfg_path=cfg_path, override=str(args.id_val_filelist))
                    if not id_val_file.exists():
                        raise FileNotFoundError(f"id_val_filelist not found: {id_val_file}")
                    best_val_file = (splits_root / f"val_id_minus_{s}.txt").resolve()
                    if not best_val_file.exists():
                        _, removed = _build_id_minus_heldout_val(
                            id_val_file=id_val_file, heldout_val_file=val_file, out_file=best_val_file
                        )
                        if removed <= 0:
                            raise RuntimeError(f"No held-out samples removed for subject={s}. Check split consistency.")
                    best_val_set = set(_read_filelist(best_val_file))
                    heldout_set = set(_read_filelist(val_file))
                    if best_val_set & heldout_set:
                        raise RuntimeError(f"best_val still contains held-out samples for subject={s}")
                    train_set = set(_read_filelist(train_file))
                    if best_val_set & train_set:
                        raise RuntimeError(f"Data leakage detected: best_val intersects train for subject={s}")
                    val_file_for_best = best_val_file
                else:
                    val_file_for_best = val_file

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
                    str(val_file_for_best),
                    "--ckpt_dir",
                    str(ckpt_dir),
                    "--resume",
                    str(args.resume),
                    "--max_steps",
                    str(int(args.max_steps)),
                ]
                if args.train_batch_size is not None:
                    cmd.extend(["--batch_size", str(int(args.train_batch_size))])
                _run_stage2_train_with_step_tqdm(cmd=cmd, env=env, cwd=mia_root, subject=str(s), max_steps=int(args.max_steps))

            if not stage2_best.exists():
                raise FileNotFoundError(f"best.pt not found: {stage2_best}")

            eval_out_dir.mkdir(parents=True, exist_ok=True)
            stage1_checkpoint = _extract_stage1_ckpt_from_stage2_best(stage2_best)
            stage1_checkpoint_aux = _extract_stage1_aux_ckpt_from_stage2_best(stage2_best)
            eval_yaml = _make_eval_yaml(
                out_dir=eval_out_dir,
                device="cuda:0",
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.eval_num_workers),
                ablation_dir=splits_root,
                train_filelist=f"train_wo_{s}.txt",
                val_filelists=[f"val_{s}.txt"],
                stage2_best=stage2_best,
                stage1_checkpoint=stage1_checkpoint,
                stage1_checkpoint_aux=stage1_checkpoint_aux,
            )
            try:
                subprocess.run([sys.executable, str(eval_script), "--config", str(eval_yaml)], check=True, env=env, cwd=str(mia_root))
            finally:
                if eval_yaml.exists():
                    eval_yaml.unlink()

            if not summary_processed.exists():
                raise FileNotFoundError(f"summary_processed.csv not found: {summary_processed}")
            avg = _parse_stage2_average(summary_processed)
            rows.append(
                {
                    "subject": s,
                    "stage2_average": f"{avg:.6f}",
                    "ckpt_dir": str(ckpt_dir),
                    "summary_processed": str(summary_processed),
                }
            )
            _append_log(run_log, f"[DONE] {s} avg={avg:.6f}")
        except Exception as e:
            _append_log(run_log, f"[ERR] {s} {type(e).__name__}: {e}")
            continue

    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(results_dir / "summary_ood_person_pose2emg.csv", rows)
    _write_summary_md(
        results_dir / "summary_ood_person_pose2emg.md",
        rows,
        title="Stage2 OOD People (Pose2EMG) Summary",
        base_cfg=base_cfg_path or (base_dir / "best.pt"),
        split_dir=split_dir,
    )
    print("[OK] out_root =", out_root)


if __name__ == "__main__":
    main()
