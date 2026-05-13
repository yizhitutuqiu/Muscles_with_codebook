import numpy as np
from pathlib import Path
temp_dir = Path("/data/litengmo/HSMR/mia_custom/custom/output/vis_infer_final_emg2pose/temp/ElbowPunch")
sample_id = "282"
cache_file = temp_dir / f"{sample_id}_pred.npy"
our_cache_file = temp_dir / f"{sample_id}_our_pred.npy"

# Find GT
for s in Path("/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val").glob(f"*/ElbowPunch/{sample_id}/joints3d.npy"):
    gt_file = s
    break

pred = np.load(cache_file)
our_pred = np.load(our_cache_file)
gt = np.load(gt_file)[:, :25, :]

print("GT root (frame 0):", gt[0, 0])
print("Official Pred root (frame 0):", pred[0, 0])
print("Our Pred root (frame 0):", our_pred[0, 0])
