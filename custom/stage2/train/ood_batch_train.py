from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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


def _discover_targets(protocol_cfg: Dict[str, Any]) -> List[str]:
    if protocol_cfg.get("targets", None):
        return [str(x).strip() for x in list(protocol_cfg["targets"]) if str(x).strip()]
    ablation_dir = Path(str(protocol_cfg["ablation_dir"])).expanduser().resolve()
    glob_pat = str(protocol_cfg.get("val_glob", "val_*.txt"))
    files = sorted([p for p in ablation_dir.glob(glob_pat) if p.is_file()])
    targets: List[str] = []
    for p in files:
        stem = p.stem
        if stem.startswith("val_"):
            t = stem[len("val_") :]
        elif stem.startswith("val"):
            t = stem[len("val") :]
        else:
            continue
        t = t.strip("_").strip()
        if t:
            targets.append(t)
    return targets


def _resolve_case_paths(protocol_cfg: Dict[str, Any], target: str) -> tuple[Path, Path]:
    ablation_dir = Path(str(protocol_cfg["ablation_dir"])).expanduser().resolve()
    train_pat = str(protocol_cfg.get("train_pattern", "train_{target}.txt"))
    val_pat = str(protocol_cfg.get("val_pattern", "val_{target}.txt"))
    train_file = (ablation_dir / train_pat.format(target=target)).resolve()
    val_file = (ablation_dir / val_pat.format(target=target)).resolve()
    return train_file, val_file


def _override_args_from_cfg(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    if overrides.get("max_steps", None) is not None:
        args += ["--max_steps", str(int(overrides["max_steps"]))]
    if overrides.get("lr", None) is not None:
        args += ["--lr", str(float(overrides["lr"]))]
    if overrides.get("batch_size", None) is not None:
        args += ["--batch_size", str(int(overrides["batch_size"]))]
    if overrides.get("save_every", None) is not None:
        args += ["--save_every", str(int(overrides["save_every"]))]
    if overrides.get("eval_every", None) is not None:
        args += ["--eval_every", str(int(overrides["eval_every"]))]
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ood_config",
        type=str,
        default="/data/litengmo/HSMR/mia_custom/custom/configs/ood/ood.yaml",
    )
    parser.add_argument("--protocol", type=str, default=None, help="Run only one protocol name.")
    parser.add_argument("--only_target", type=str, default=None, help="Run only one target (exercise or SubjectX).")
    parser.add_argument("--list_cases", type=_str2bool, default=False, help="If true, print 'protocol<TAB>target' per case and exit.")
    parser.add_argument("--dry_run", type=_str2bool, default=False)
    parser.add_argument("--resume", type=_str2bool, default=None, help="Override train.resume in yaml.")
    args = parser.parse_args()

    cfg_path = Path(args.ood_config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"OOD config not found: {cfg_path}")
    cfg = _load_yaml(cfg_path)

    base_cfg = Path(str(cfg.get("base", {}).get("stage2_config", ""))).expanduser().resolve()
    if not base_cfg.exists():
        raise FileNotFoundError(f"Base stage2 config not found: {base_cfg}")

    ckpt_root = Path(str(cfg.get("output", {}).get("ckpt_root", ""))).expanduser().resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    runtime = cfg.get("runtime", {}) or {}
    device = str(runtime.get("device", "cuda")).strip()

    train_cfg = cfg.get("train", {}) or {}
    resume = bool(train_cfg.get("resume", True)) if args.resume is None else bool(args.resume)
    init_mode = str(train_cfg.get("init", "warm_start")).strip().lower()
    init_ckpt = train_cfg.get("init_stage2_checkpoint", None)
    overrides = train_cfg.get("overrides", {}) or {}
    override_args = _override_args_from_cfg(overrides)

    train_script = (Path(__file__).resolve().parent / "train_stage2_pose2emg.py").resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"train_stage2_pose2emg.py not found: {train_script}")

    protocols = cfg.get("protocols", []) or []
    cases: List[tuple[str, str, Path, Path, Path]] = []
    for p in protocols:
        if not isinstance(p, dict) or not bool(p.get("enabled", True)):
            continue
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        if args.protocol is not None and str(args.protocol).strip() != name:
            continue

        targets = _discover_targets(p)
        if args.only_target is not None:
            ot = str(args.only_target).strip()
            targets = [t for t in targets if t == ot]

        if not targets:
            continue

        for target in targets:
            train_file, val_file = _resolve_case_paths(p, target)
            if not train_file.exists():
                raise FileNotFoundError(f"Missing train filelist: {train_file}")
            if not val_file.exists():
                raise FileNotFoundError(f"Missing val filelist: {val_file}")

            out_dir = (ckpt_root / name / target).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            cases.append((name, target, train_file, val_file, out_dir))

    if bool(args.list_cases):
        for name, target, _, _, _ in cases:
            print(f"{name}\t{target}")
        return

    for name, target, train_file, val_file, out_dir in cases:
        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            str(base_cfg),
            "--device",
            device,
            "--train_filelist",
            str(train_file),
            "--val_filelist",
            str(val_file),
            "--ckpt_dir",
            str(out_dir),
            "--resume",
            "true" if resume else "false",
            *override_args,
        ]
        if init_mode == "warm_start" and init_ckpt:
            cmd += ["--init_stage2_checkpoint", str(init_ckpt)]

        print("[OODBatchTrain]", {"protocol": name, "target": target, "cmd": " ".join(cmd)})
        if args.dry_run:
            continue

        env = dict(os.environ)
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
