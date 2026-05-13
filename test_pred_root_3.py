import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path("/data/litengmo/HSMR/mia_custom")
sys.path.insert(0, str(REPO_ROOT))

from custom.tools.Mia_style_eval import _build_stage2_from_ckpt
import torch

ckpt_path = Path("/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_h8_baseline/best.pt")
device = torch.device("cpu")
model, cfg, _, _ = _build_stage2_from_ckpt(ckpt_path, device, stage1_override_ckpt=None)

sample_id = "282"
for s in Path("/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val").glob(f"*/ElbowPunch/{sample_id}"):
    emg = np.load(s / "emgvalues.npy").astype(np.float32)
    gt = np.load(s / "joints3d.npy")[:, :25, :]
    break

inputs = torch.from_numpy(emg.T).unsqueeze(0).to(device) # B, T, 8
cond = torch.tensor([[0.8]], dtype=torch.float32, device=device)
with torch.no_grad():
    out = model(inputs, cond=cond)
    pred = out["pred"].squeeze(0).numpy()

print("Model raw pred root (frame 0):", pred[0, 0])
print("GT root (frame 0):", gt[0, 0])
