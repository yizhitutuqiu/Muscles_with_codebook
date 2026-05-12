import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')

# X: left/right (red), Z: forward/backward (green), Y: up/down (blue)
ax.plot([0, 1], [0, 0], [0, 0], color='r', label='X (L/R)')
ax.plot([0, 0], [0, 1], [0, 0], color='g', label='Z (F/B)')
ax.plot([0, 0], [0, 0], [0, 1], color='b', label='Y (U/D)')

ax.set_xlabel('X_plot (X)')
ax.set_ylabel('Y_plot (Z)')
ax.set_zlabel('Z_plot (Y)')

# Try different azim
# If we want to look straight at the person (assuming person faces positive Z or negative Z)
# If person faces positive Z (green), we want to look from positive Z towards origin -> camera at +Z.
# If person faces negative Z, we look from -Z towards origin.

ax.view_init(0, -90)
plt.legend()
plt.savefig("test_view_-90.png")

ax.view_init(0, 90)
plt.savefig("test_view_90.png")

ax.view_init(0, 0)
plt.savefig("test_view_0.png")

ax.view_init(0, 180)
plt.savefig("test_view_180.png")
