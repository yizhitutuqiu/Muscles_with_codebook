import sys
sys.path.insert(0, '/data/litengmo/HSMR/mia_custom')
from custom.tools.Mia_style_eval import _build_stage2_from_ckpt
import torch

ckpt = '/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_h8_baseline/best.pt'
model, _, _, _ = _build_stage2_from_ckpt(ckpt, torch.device('cpu'), stage1_override_ckpt=None)
print("Model loaded successfully")
