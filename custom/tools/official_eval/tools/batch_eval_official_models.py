#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _ensure_pythonpath(mia_root: Path) -> None:
    extra = [str(mia_root), str((mia_root / "musclesinaction").resolve())]
    for p in reversed(extra):
        if p not in sys.path:
            sys.path.insert(0, p)


def _find_ckpt(exp_dir: Path) -> Optional[Path]:
    for name in ["model_100.pth", "latestcheckpoint.pth", "checkpoint.pth"]:
        p = exp_dir / name
        if p.exists():
            return p
    return None


@dataclass(frozen=True)
class Case:
    name: str
    train_filelist: Path
    val_filelist: Path


def _discover_id_cases(mia_root: Path) -> List[Case]:
    ablation_dir = Path(
        "/data/litengmo/HSMR/golf_third_party/musclesinaction/musclesinaction/ablation/generalization_ID_nocond_exercises"
    )
    train_file = ablation_dir / "train.txt"
    val_files = sorted([p for p in ablation_dir.glob("val*.txt") if p.is_file()])
    return [Case(name=p.stem, train_filelist=train_file, val_filelist=p) for p in val_files]


def _discover_ood_cases(split_dir: Path) -> List[Case]:
    val_files = sorted([p for p in split_dir.glob("val_*.txt") if p.is_file()])
    cases: List[Case] = []
    for vf in val_files:
        ex = vf.stem.replace("val_", "", 1).strip()
        tf = split_dir / f"train_{ex}.txt"
        cases.append(Case(name=ex, train_filelist=tf, val_filelist=vf))
    return cases


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _write_table(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _eval_model(
    *,
    model: Any,
    filelist: Path,
    cond: bool,
    task: str,
    device: Any,
    batch_size: int,
    num_workers: int,
    step: int,
    std: bool,
) -> float:
    from custom.tools.Mia_style_eval import _build_loader, _compute_official_metrics, _eval_official_model_on_filelist

    loader = _build_loader(
        filelist_path=filelist,
        split="val",
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        step=int(step),
        std=bool(std),
        cond=bool(cond),
        percent=1.0,
        max_samples=None,
        device=device,
    )
    pred_cat, gt_cat, _ = _eval_official_model_on_filelist(model=model, loader=loader, device=device, task=task)
    metrics = _compute_official_metrics(pred_cat, gt_cat, task=task)
    return float(metrics["official_global_rmse"])


def _load_official(ckpt: Path, *, device: Any, task: str) -> Any:
    from custom.tools.Mia_style_eval import _load_official_model

    model, _, _ = _load_official_model(ckpt, device=device, task=task)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/eval_results",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--std", type=str, default="False")
    parser.add_argument(
        "--pose2emg_official_cond",
        type=str,
        default="pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth",
    )
    parser.add_argument(
        "--pose2emg_official_nocond",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/output/20260410_102342/checkpoints/generalization_new_nocond_clean_posetoemg/model_100.pth",
    )
    parser.add_argument(
        "--emg2pose_official_cond",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/output/20260510_000126/checkpoints/official_reproduction_cond_emgtopose_threed/model_100.pth",
    )
    parser.add_argument(
        "--ood_split_dir",
        type=str,
        default="/data/litengmo/HSMR/golf_third_party/musclesinaction/musclesinaction/ablation/generalizationexercises",
    )
    parser.add_argument(
        "--pose2emg_ood_run_dir",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/ood/posetoemg_ood_20260514_104627",
    )
    parser.add_argument(
        "--emg2pose_ood_run_dir",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/ood/emgtopose_ood_20260514_111255",
    )
    parser.add_argument("--only", type=str, default="", help="Comma-separated case names to run (e.g. ElbowPunch,Running).")
    args = parser.parse_args()

    mia_root = _repo_root()
    _ensure_pythonpath(mia_root)

    import torch

    device = torch.device(str(args.device))
    out_dir = _resolve(args.out_dir, mia_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = [x.strip() for x in str(args.only).split(",") if x.strip()]
    only_set = set(only)

    id_cases = _discover_id_cases(mia_root)
    if only_set:
        id_cases = [c for c in id_cases if c.name.replace("val", "").replace(".txt", "") in only_set or c.name in only_set]

    ood_split_dir = Path(str(args.ood_split_dir)).expanduser().resolve()
    ood_cases = _discover_ood_cases(ood_split_dir)
    if only_set:
        ood_cases = [c for c in ood_cases if c.name in only_set]

    pose2emg_cond_ckpt = _resolve(args.pose2emg_official_cond, mia_root)
    pose2emg_nocond_ckpt = _resolve(args.pose2emg_official_nocond, mia_root)
    emg2pose_cond_ckpt = _resolve(args.emg2pose_official_cond, mia_root)

    pose2emg_cond_model = _load_official(pose2emg_cond_ckpt, device=device, task="pose2emg")
    pose2emg_nocond_model = _load_official(pose2emg_nocond_ckpt, device=device, task="pose2emg")
    emg2pose_cond_model = _load_official(emg2pose_cond_ckpt, device=device, task="emg2pose")

    std_flag = str(args.std).strip().lower() in ("1", "true", "t", "yes", "y", "on")

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None

    pose2emg_id_rows: List[List[Any]] = []
    pose2emg_id_vals_cond: List[float] = []
    pose2emg_id_vals_nocond: List[float] = []
    it = id_cases
    if tqdm is not None:
        it = tqdm(id_cases, desc="official pose2emg (ID)", dynamic_ncols=True)
    for case in it:
        v1 = _eval_model(
            model=pose2emg_cond_model,
            filelist=case.val_filelist,
            cond=True,
            task="pose2emg",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        v2 = _eval_model(
            model=pose2emg_nocond_model,
            filelist=case.val_filelist,
            cond=False,
            task="pose2emg",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        ex = case.val_filelist.stem.replace("val", "").replace(".txt", "").strip("_").strip()
        pose2emg_id_rows.append([ex, f"{v1:.6f}", f"{v2:.6f}"])
        pose2emg_id_vals_cond.append(float(v1))
        pose2emg_id_vals_nocond.append(float(v2))
    pose2emg_id_rows.append(["Average", f"{_mean(pose2emg_id_vals_cond):.6f}", f"{_mean(pose2emg_id_vals_nocond):.6f}"])
    _write_table(out_dir / "pose2emg_id.csv", ["exercise", "official_cond", "official_nocond"], pose2emg_id_rows)

    emg2pose_id_rows: List[List[Any]] = []
    emg2pose_id_vals_cond: List[float] = []
    emg2pose_id_vals_nocond: List[float] = []
    it2 = id_cases
    if tqdm is not None:
        it2 = tqdm(id_cases, desc="official emg2pose (ID)", dynamic_ncols=True)
    for case in it2:
        v_cond = _eval_model(
            model=emg2pose_cond_model,
            filelist=case.val_filelist,
            cond=True,
            task="emg2pose",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        v_nocond = _eval_model(
            model=emg2pose_cond_model,
            filelist=case.val_filelist,
            cond=False,
            task="emg2pose",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        ex = case.val_filelist.stem.replace("val", "").replace(".txt", "").strip("_").strip()
        emg2pose_id_rows.append([ex, f"{v_cond:.6f}", f"{v_nocond:.6f}"])
        emg2pose_id_vals_cond.append(float(v_cond))
        emg2pose_id_vals_nocond.append(float(v_nocond))
    emg2pose_id_rows.append(["Average", f"{_mean(emg2pose_id_vals_cond):.6f}", f"{_mean(emg2pose_id_vals_nocond):.6f}"])
    _write_table(out_dir / "emg2pose_id.csv", ["exercise", "official_cond_mpjpe", "official_nocond_mpjpe"], emg2pose_id_rows)

    pose2emg_ood_rows: List[List[Any]] = []
    pose2emg_ood_vals_cond: List[float] = []
    pose2emg_ood_vals_nocond: List[float] = []
    pose2emg_ood_dir = Path(str(args.pose2emg_ood_run_dir)).expanduser().resolve()
    ckpt_root = pose2emg_ood_dir / "checkpoints"
    it3 = ood_cases
    if tqdm is not None:
        it3 = tqdm(ood_cases, desc="official pose2emg (OOD)", dynamic_ncols=True)
    for case in it3:
        exp_dir = ckpt_root / f"official_ood_cond_{case.name}_posetoemg"
        ckpt = _find_ckpt(exp_dir)
        if ckpt is None:
            continue
        model = _load_official(ckpt, device=device, task="pose2emg")
        v_cond = _eval_model(
            model=model,
            filelist=case.val_filelist,
            cond=True,
            task="pose2emg",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        v_nocond = _eval_model(
            model=model,
            filelist=case.val_filelist,
            cond=False,
            task="pose2emg",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        pose2emg_ood_rows.append([case.name, f"{v_cond:.6f}", f"{v_nocond:.6f}", str(ckpt)])
        pose2emg_ood_vals_cond.append(float(v_cond))
        pose2emg_ood_vals_nocond.append(float(v_nocond))
    if pose2emg_ood_vals_cond:
        pose2emg_ood_rows.append(
            ["Average", f"{_mean(pose2emg_ood_vals_cond):.6f}", f"{_mean(pose2emg_ood_vals_nocond):.6f}", ""]
        )
    _write_table(out_dir / "pose2emg_ood.csv", ["exercise", "official_cond", "official_nocond", "checkpoint"], pose2emg_ood_rows)

    emg2pose_ood_rows: List[List[Any]] = []
    emg2pose_ood_vals_cond: List[float] = []
    emg2pose_ood_vals_nocond: List[float] = []
    emg2pose_ood_dir = Path(str(args.emg2pose_ood_run_dir)).expanduser().resolve()
    ckpt_root2 = emg2pose_ood_dir / "checkpoints"
    it4 = ood_cases
    if tqdm is not None:
        it4 = tqdm(ood_cases, desc="official emg2pose (OOD)", dynamic_ncols=True)
    for case in it4:
        exp_dir = ckpt_root2 / f"official_ood_cond_{case.name}_emgtopose"
        ckpt = _find_ckpt(exp_dir)
        if ckpt is None:
            continue
        model = _load_official(ckpt, device=device, task="emg2pose")
        v_cond = _eval_model(
            model=model,
            filelist=case.val_filelist,
            cond=True,
            task="emg2pose",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        v_nocond = _eval_model(
            model=model,
            filelist=case.val_filelist,
            cond=False,
            task="emg2pose",
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            step=int(args.step),
            std=std_flag,
        )
        emg2pose_ood_rows.append([case.name, f"{v_cond:.6f}", f"{v_nocond:.6f}", str(ckpt)])
        emg2pose_ood_vals_cond.append(float(v_cond))
        emg2pose_ood_vals_nocond.append(float(v_nocond))
    if emg2pose_ood_vals_cond:
        emg2pose_ood_rows.append(
            ["Average", f"{_mean(emg2pose_ood_vals_cond):.6f}", f"{_mean(emg2pose_ood_vals_nocond):.6f}", ""]
        )
    _write_table(out_dir / "emg2pose_ood.csv", ["exercise", "official_cond_mpjpe", "official_nocond_mpjpe", "checkpoint"], emg2pose_ood_rows)

    print("[OK] out_dir =", out_dir)


if __name__ == "__main__":
    main()
