#!/usr/bin/env python
import argparse
import copy
import csv
import os
import subprocess
from pathlib import Path
import yaml


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--list_cases", action="store_true")
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        batch_cfg = yaml.safe_load(f)

    base_cfg_path = batch_cfg["base_config"]
    if not Path(base_cfg_path).is_absolute():
        base_cfg_path = Path(cfg_path.parent / ".." / ".." / ".." / base_cfg_path).resolve()
    
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    checkpoint_dir = batch_cfg.get("checkpoint_dir", "custom/checkpoints/batch_codebook")
    csv_output = batch_cfg.get("csv_output", "custom/checkpoints/batch_codebook/summary.csv")
    
    experiments = batch_cfg.get("experiments", [])
    
    if args.list_cases:
        for exp in experiments:
            print(exp["name"])
        return

    if args.summary:
        print("[Summary] Gathering results...")
        all_results = []
        for exp in experiments:
            name = exp["name"]
            ckpt_path = Path(checkpoint_dir) / name
            metrics_file = ckpt_path / "best_metrics.csv"
            
            res = {"name": name, "val_loss": float('inf'), "step": "-", "activation_rate": "-", 
                   "j3d_self_smooth_l1": "-", "emg_self_smooth_l1": "-", 
                   "emg_to_j3d_smooth_l1": "-", "j3d_to_emg_smooth_l1": "-"}
            
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    # 找到 val_loss 最小的那一行 (以防保存了多次)
                    best_row = None
                    for row in reader:
                        if best_row is None or float(row["val_loss"]) < float(best_row["val_loss"]):
                            best_row = row
                    
                    if best_row:
                        for k in best_row:
                            if k in res:
                                res[k] = best_row[k]
                                
            all_results.append(res)
            
        all_results.sort(key=lambda x: float(x["val_loss"]))
        
        csv_path = Path(csv_output).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[Summary] Wrote summary to {csv_path}")
        return

    if args.case:
        exp = next((e for e in experiments if e["name"] == args.case), None)
        if exp is None:
            raise ValueError(f"Case {args.case} not found in config.")
        
        # Merge config
        run_cfg = copy.deepcopy(base_cfg)
        _deep_update(run_cfg, exp)
        
        # 自动同步 code_dim 到所有 modality，防止 FrameCodebookModel 校验报错
        if "model" in exp and "vq" in exp["model"] and "code_dim" in exp["model"]["vq"]:
            new_code_dim = exp["model"]["vq"]["code_dim"]
            if "modalities" in run_cfg.get("model", {}):
                for mod_name, mod_cfg in run_cfg["model"]["modalities"].items():
                    if isinstance(mod_cfg, dict):
                        mod_cfg["code_dim"] = new_code_dim
        
        run_cfg["experiment"]["name"] = exp["name"]
        run_cfg["checkpoints"]["dir"] = str(Path(checkpoint_dir) / exp["name"])
        
        # Temp save run config
        tmp_cfg_path = Path(cfg_path.parent / f"temp_run_{exp['name']}.yaml").resolve()
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)
            
        print(f"[Batch] Running case: {exp['name']} on {args.device}")
        cmd = [
            "python", "custom/train/train_frame_codebook.py",
            "--config", str(tmp_cfg_path),
            "--device", args.device
        ]
        try:
            subprocess.run(cmd, check=True)
        finally:
            if tmp_cfg_path.exists():
                tmp_cfg_path.unlink()

if __name__ == "__main__":
    main()