import argparse
import os
import subprocess
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="Path to the batch checkpoint directory (e.g. custom/stage2/checkpoints/batch_stage2_h5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.exists() or not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    # Find all experiment directories in the batch dir
    exp_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and (d / "best.pt").exists()]
    if not exp_dirs:
        print(f"No valid experiment directories with best.pt found in {batch_dir}")
        return

    # Base eval template config
    base_eval_cfg = {
        "runtime": {"device": args.device, "num_workers": 4, "batch_size": 8},
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
        "output": {}
    }

    import copy

    # Evaluate each experiment
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        print(f"\n{'='*50}\nEvaluating experiment: {exp_name}\n{'='*50}")
        
        # We need to construct a specific config for this experiment
        ckpt_path = exp_dir / "best.pt"
        
        # Load the training config to get stage1 path if possible
        train_cfg_path = exp_dir / "config.yaml"
        stage1_ckpt = ""
        if train_cfg_path.exists():
            with open(train_cfg_path, "r") as f:
                t_cfg = yaml.safe_load(f)
                if t_cfg and "stage1" in t_cfg:
                    stage1_ckpt = t_cfg["stage1"].get("checkpoint", "")

        eval_cfg = copy.deepcopy(base_eval_cfg)
        eval_cfg["output"] = {"out_dir": str(exp_dir / "eval_results")}
        eval_cfg["methods"] = {
            "stage2": {
                "enabled": True,
                "checkpoint": str(ckpt_path),
                "stage1_checkpoint": stage1_ckpt
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
        
        # Save temp config
        temp_cfg_path = exp_dir / "temp_eval.yaml"
        with open(temp_cfg_path, "w") as f:
            yaml.dump(eval_cfg, f, default_flow_style=False)
            
        # Run eval
        cmd = [
            "python", "custom/tools/Mia_style_eval.py",
            "--config", str(temp_cfg_path)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        
        # Cleanup
        if temp_cfg_path.exists():
            temp_cfg_path.unlink()
            
        print(f"Evaluation finished for {exp_name}. Results saved to {exp_dir / 'eval_results'}")

if __name__ == "__main__":
    main()