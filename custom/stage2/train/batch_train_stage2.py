import argparse
import copy
import csv
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


def _run_post_training_eval(out_dir: str, device: str) -> None:
    """Run Mia_style_eval.py automatically on the newly trained model"""
    import subprocess
    from pathlib import Path
    import yaml
    
    out_path = Path(out_dir)
    best_ckpt = out_path / "best.pt"
    if not best_ckpt.exists():
        print(f"Skipping evaluation: {best_ckpt} not found.")
        return
        
    print(f"\n{'='*50}\nStarting Post-Training Evaluation for {out_path.name}\n{'='*50}")
    
    # Base eval template config
    base_eval_cfg = {
        "runtime": {"device": device, "num_workers": 4, "batch_size": 8},
        "data": {"step": 30, "std": False, "percent": 1.0, "joints3d_root_center": True},
        "protocols": [
            {
                "name": "id_exercises",
                "enabled": True,
                "split": "val",
                "ablation_dir": "/data/litengmo/HSMR/golf_third_party/musclesinaction/musclesinaction/ablation/generalization_ID_nocond_exercises",
                "train_filelist": "train.txt",
                "val_glob": "val*.txt"
            }
        ],
        "output": {"out_dir": str(out_path / "eval_results")},
        "methods": {
            "stage2": {
                "enabled": True,
                "checkpoint": str(best_ckpt),
            },
            "official_cond": {
                "enabled": True,
                "checkpoint": "pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth"
            },
            "official_nocond": {
                "enabled": True,
                "checkpoint": "/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/output/20260410_102342/checkpoints/generalization_new_nocond_clean_posetoemg/model_100.pth"
            },
            "retrieval": {
                "enabled": False
            }
        }
    }
    
    # Check if we need stage1 checkpoint explicitly
    train_cfg_path = out_path / "config.yaml"
    if train_cfg_path.exists():
        with open(train_cfg_path, "r") as f:
            t_cfg = yaml.safe_load(f)
            if t_cfg and "stage1" in t_cfg:
                base_eval_cfg["methods"]["stage2"]["stage1_checkpoint"] = t_cfg["stage1"].get("checkpoint", "")
                
    # Save temp eval config
    temp_cfg_path = out_path / "temp_eval.yaml"
    with open(temp_cfg_path, "w") as f:
        yaml.dump(base_eval_cfg, f, default_flow_style=False)
        
    cmd = [
        "python", "custom/tools/Mia_style_eval.py",
        "--config", str(temp_cfg_path)
    ]
    
    subprocess.run(cmd, check=False)
    if temp_cfg_path.exists():
        temp_cfg_path.unlink()
    print(f"Evaluation finished. Results saved to {out_path / 'eval_results'}")


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

    checkpoint_dir = batch_cfg.get("checkpoint_dir", "custom/stage2/checkpoints/batch_stage2")
    csv_output = batch_cfg.get("csv_output", "custom/stage2/checkpoints/batch_stage2/summary.csv")

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

            # Stage2 评测指标默认格式: step, loss_smooth_l1, loss_mse, loss_mae, lr, fps... 
            # 及各通道误差等。这里抓取 step 和 validation loss
            res = {"name": name, "val_smooth_l1": float('inf'), "val_rmse": float('inf'), "step": "-"}

            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    best_row = None
                    # Stage 2 eval output usually contains 'val_smooth_l1', 'val_rmse'
                    for row in reader:
                        # 查找 val_rmse 最小的一行，如果不存在则按 smooth_l1 找
                        metric_key = "val_rmse" if "val_rmse" in row else "val_smooth_l1"
                        if best_row is None or float(row.get(metric_key, float('inf'))) < float(best_row.get(metric_key, float('inf'))):
                            best_row = row

                    if best_row:
                        for k in best_row:
                            if k in res:
                                res[k] = best_row[k]
                            # 把其他键也全带上
                            else:
                                res[k] = best_row[k]

            all_results.append(res)

        all_results.sort(key=lambda x: float(x.get("val_rmse", x.get("val_smooth_l1", float('inf')))))

        csv_path = Path(csv_output).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if len(all_results) > 0:
            fieldnames = list(all_results[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
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

        run_cfg["experiment"]["name"] = exp["name"]
        run_cfg.setdefault("checkpoints", {})["dir"] = str(Path(checkpoint_dir) / exp["name"])
        
        # 强制单卡运行
        run_cfg.setdefault("runtime", {})
        run_cfg["runtime"].setdefault("distributed", {})
        run_cfg["runtime"]["distributed"]["enabled"] = False
        run_cfg["runtime"]["device"] = args.device

        # Temp save run config
        tmp_cfg_path = Path(cfg_path.parent / f"temp_run_stage2_{exp['name']}.yaml").resolve()
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)

        print(f"[Batch] Running case: {exp['name']} on {args.device}")
        cmd = [
            "python", "custom/stage2/train/train_stage2_pose2emg.py",
            "--config", str(tmp_cfg_path),
            "--device", args.device
        ]
        try:
            subprocess.run(cmd, check=True)
            
            # Trigger automatic post-training evaluation
            out_dir = str(Path(checkpoint_dir) / exp["name"])
            try:
                _run_post_training_eval(out_dir, args.device)
            except Exception as e:
                print(f"Error during post-training evaluation: {e}")

        finally:
            if tmp_cfg_path.exists():
                tmp_cfg_path.unlink()

if __name__ == "__main__":
    main()
