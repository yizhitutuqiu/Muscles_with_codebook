import numpy as np
joints_t = np.load('/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val/Subject0/ElbowPunch/188/joints3d.npy')

# Compute global bounds across time and 25 joints
joints = joints_t[:, :25, :]
x = joints[..., 0]
y = joints[..., 2] # Plotted on Y
z = joints[..., 1] # Plotted on Z

x_mid = (x.max() + x.min()) / 2
y_mid = (y.max() + y.min()) / 2
z_mid = (z.max() + z.min()) / 2

max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0

print(f"Midpoints: X={x_mid:.3f}, Y={y_mid:.3f}, Z={z_mid:.3f}")
print(f"Max range: {max_range:.3f}")
print(f"X limits: {x_mid - max_range:.3f} to {x_mid + max_range:.3f}")
print(f"Y limits: {y_mid - max_range:.3f} to {y_mid + max_range:.3f}")
print(f"Z limits: {z_mid - max_range:.3f} to {z_mid + max_range:.3f}")

