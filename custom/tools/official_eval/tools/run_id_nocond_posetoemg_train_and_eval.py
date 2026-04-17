#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    # .../mia_custom/custom/tools/official_eval/tools/<this_file>
    return Path(__file__).resolve().parents[4]


def _ensure_importable() -> None:
    try:
        from benedict import benedict  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Official scripts require python-benedict (NOT the 'benedict' 0.3.2 package).\n"
            "Fix in your active conda env:\n"
            "  python -m pip uninstall -y benedict\n"
            "  python -m pip install \"python-benedict[yaml]\"\n"
            f"Original import error: {e}"
        )


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    no_capture_output: bool,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        if not no_capture_output:
            subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            return
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _find_trained_ckpt(ckpt_dir: Path, *, name: str) -> Path:
    exp_dir = ckpt_dir / name
    p100 = exp_dir / "model_100.pth"
    if p100.exists():
        return p100
    latest = exp_dir / "latestcheckpoint.pth"
    if latest.exists():
        return latest
    raise FileNotFoundError(f"Expected model_100.pth or latestcheckpoint.pth under {exp_dir}")


def _parse_official_rmse_from_log(log_file: Path) -> float:
    txt = log_file.read_text(encoding="utf-8", errors="ignore")
    # The official script prints: tensor(12.3456) or tensor(12.837, dtype=torch.float64)
    ms = re.findall(r"tensor\(([-+0-9.eE]+)(?:,|\))", txt)
    if not ms:
        raise RuntimeError(f"Cannot find printed RMSE tensor(...) in log: {log_file}")
    return float(ms[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train official ID-nocond pose->EMG and evaluate with official metrics.")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA_VISIBLE_DEVICES for training/eval. Default: 0")
    parser.add_argument("--num_epochs", type=int, default=101, help="Official train.py uses 0-based epoch index; 101 gives model_100.pth.")
    parser.add_argument("--train_filelist", type=str, default="musclesinaction/ablation/generalization_ID_nocond_exercises/train.txt")
    parser.add_argument("--val_for_selection", type=str, default="musclesinaction/ablation/generalization_ID_nocond_exercises/valRunning.txt")
    parser.add_argument("--eval_dir", type=str, default="musclesinaction/ablation/generalization_ID_nocond_exercises")
    parser.add_argument("--name", type=str, default="generalization_new_nocond_clean_posetoemg", help="Experiment name (checkpoint subdir name).")
    parser.add_argument("--batch_size_eval", type=int, default=8)
    parser.add_argument("--num_workers_eval", type=int, default=4)
    parser.add_argument("--no_capture_output", action="store_true", help="Stream subprocess output to terminal (also saves logs).")
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root directory. Default: custom/tools/official_eval/output/<timestamp>/",
    )
    args = parser.parse_args()

    _ensure_importable()

    root = _repo_root()
    mia_root = root
    if not (mia_root / "musclesinaction").exists() or not (mia_root / "custom").exists():
        raise FileNotFoundError(
            f"Expected mia_custom repo root containing 'musclesinaction/' and 'custom/' at: {mia_root}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_root:
        out_root = Path(args.out_root).expanduser().resolve()
    else:
        out_root = (mia_root / "custom" / "tools" / "official_eval" / "output" / stamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_root = (out_root / "checkpoints").resolve()
    logs_root = (out_root / "logs").resolve()
    eval_root = (out_root / "eval").resolve()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    # Official scripts mix absolute imports (musclesinaction.*) and repo-root-relative imports (vis.*, pipeline, ...).
    # Make both `mia_custom/` and `mia_custom/musclesinaction/` importable.
    extra_paths = [str(mia_root), str((mia_root / "musclesinaction").resolve())]
    env["PYTHONPATH"] = ":".join(extra_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["WANDB_SILENT"] = "true"
    env["MIA_DISABLE_RENDERER"] = "1"
    env["MIA_DISABLE_PDB"] = "1"
    env["MPLBACKEND"] = "Agg"

    train_cmd = [
        sys.executable,
        "musclesinaction/train.py",
        "--name",
        str(args.name),
        "--predemg",
        "True",
        "--threed",
        "True",
        "--cond",
        "False",
        "--std",
        "False",
        "--num_epochs",
        str(int(args.num_epochs)),
        "--checkpoint_path",
        str(ckpt_root),
        "--log_path",
        str(logs_root),
        "--data_path_train",
        str(args.train_filelist),
        "--data_path_val",
        str(args.val_for_selection),
        "--device",
        "cuda",
    ]
    _run(
        train_cmd,
        cwd=mia_root,
        env=env,
        log_file=logs_root / "train_nocond_posetoemg.log",
        no_capture_output=bool(args.no_capture_output),
    )

    trained_ckpt = _find_trained_ckpt(ckpt_root, name=str(args.name))

    eval_dir = (mia_root / str(args.eval_dir)).resolve()
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")
    val_lists = sorted(eval_dir.glob("val*.txt"))
    if not val_lists:
        raise FileNotFoundError(f"No val*.txt found under: {eval_dir}")

    rows: list[dict[str, str]] = []
    for fp in val_lists:
        ex_name = fp.stem
        out_dir = (eval_root / ex_name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_log = logs_root / f"eval_{ex_name}.log"
        eval_cmd = [
            sys.executable,
            "musclesinaction/inference_scripts/inference_id_transf_cond_exercises_posetoemg.py",
            "--name",
            str(args.name),
            "--std",
            "False",
            "--threed",
            "True",
            "--predemg",
            "True",
            "--cond",
            "False",
            "--resume",
            str(trained_ckpt),
            "--checkpoint_path",
            str(ckpt_root),
            "--log_path",
            str(logs_root),
            "--data_path_train",
            str(args.train_filelist),
            "--data_path_val",
            str(fp),
            "--device",
            "cuda",
        ]
        _run(
            eval_cmd,
            cwd=mia_root,
            env=env,
            log_file=eval_log,
            no_capture_output=bool(args.no_capture_output),
        )
        rmse = _parse_official_rmse_from_log(eval_log)
        rows.append(
            {
                "exercise": ex_name,
                "official_global_rmse": f"{rmse:.6f}",
                "eval_log": str(eval_log),
            }
        )

    if rows:
        def _k(x: dict[str, str]) -> float:
            try:
                return float(x.get("official_global_rmse", "inf"))
            except Exception:
                return float("inf")

        rows.sort(key=_k)
        merged = eval_root / "summary_all.csv"
        with open(merged, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("[OK] trained_ckpt =", trained_ckpt)
    print("[OK] out_root =", out_root)


if __name__ == "__main__":
    main()
