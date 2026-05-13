import sys
from pathlib import Path
import torch

REPO_ROOT = Path("/data/litengmo/HSMR/mia_custom")
sys.path.insert(0, str(REPO_ROOT))

from custom.tools.Mia_style_eval import _build_stage2_from_ckpt

ckpt_path = Path("/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_h8_baseline/best.pt")
device = torch.device("cpu")

try:
    model, cfg, stage1_path, stage1 = _build_stage2_from_ckpt(ckpt_path, device, stage1_override_ckpt=None)
    print("Success loading our emg2pose model")
except Exception as e:
    print(f"Failed: {e}")

ckpt_path_2 = Path("/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/batch_stage2_h8_guided_moe/exp_h8_baseline_8exp/best.pt")
try:
    model2, cfg2, stage1_path2, stage1_2 = _build_stage2_from_ckpt(ckpt_path_2, device, stage1_override_ckpt=None)
    print("Success loading our pose2emg model")
except Exception as e:
    print(f"Failed: {e}")
