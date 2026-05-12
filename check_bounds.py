import numpy as np
joints = np.load('/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val/Subject0/ElbowPunch/188/joints3d.npy')
print("Shape:", joints.shape)
print("X range:", joints[:, :, 0].min(), joints[:, :, 0].max())
print("Y range:", joints[:, :, 1].min(), joints[:, :, 1].max())
print("Z range:", joints[:, :, 2].min(), joints[:, :, 2].max())
