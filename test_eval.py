import torch
import numpy as np

gt = np.load('/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val/Subject0/ElbowPunch/188/joints3d.npy')
print("GT raw root:", gt[0, 0])
