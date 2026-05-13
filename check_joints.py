import numpy as np

gt = np.load('/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val/Subject0/ElbowPunch/188/joints3d.npy')

# Check first frame
frame = gt[0]
print("Joint 0:", frame[0])
print("Joint 8:", frame[8])  # often torso/neck
print("Joint 15:", frame[15]) # often head

# Find the joint with max Y (highest) and min Y (lowest)
# Note: SMPL usually has Y up or down depending on coordinate system. 
# In _render_skeleton_panel, minvaly and maxvaly use index 2 (Z axis?) wait, let's check:
# ax.scatter3D(gt[j, 0], gt[j, 2], gt[j, 1]) -> x=0, y=2, z=1
# Let's print min/max in index 1 and 2
y_vals = frame[:, 1]
z_vals = frame[:, 2]
print(f"Max index 1: {np.argmax(y_vals)} (val={y_vals.max():.3f}), Min index 1: {np.argmin(y_vals)} (val={y_vals.min():.3f})")
print(f"Max index 2: {np.argmax(z_vals)} (val={z_vals.max():.3f}), Min index 2: {np.argmin(z_vals)} (val={z_vals.min():.3f})")
