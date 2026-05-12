import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Add _compute_3d_bounds
bounds_code = """
def _compute_3d_bounds(gt_joints: np.ndarray, pred_joints: np.ndarray) -> dict:
    all_joints = np.concatenate([gt_joints, pred_joints], axis=0)
    x = all_joints[..., 0]
    y = all_joints[..., 2]
    z = all_joints[..., 1]
    
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2
    
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0
    max_range *= 1.1 # 10% padding
    
    return {
        'x_min': x_mid - max_range, 'x_max': x_mid + max_range,
        'y_min': y_mid - max_range, 'y_max': y_mid + max_range,
        'z_min': z_mid - max_range, 'z_max': z_mid + max_range,
    }

def _render_skeleton_panel(joints3d_25_3: np.ndarray, width: int, height: int, title: str, bounds: dict = None) -> np.ndarray:"""

content = re.sub(r'def _render_skeleton_panel\(joints3d_25_3: np.ndarray, width: int, height: int, title: str\) -> np.ndarray:', bounds_code, content)

old_limits = """    minvaly, maxvaly = np.min(joints3d_25_3[:, 2]), np.max(joints3d_25_3[:, 2])
    minvalz, maxvalz = np.min(joints3d_25_3[:, 1]), np.max(joints3d_25_3[:, 1])

    ax.set_xlim3d([-1.0, 2.0])
    ax.set_zlim3d([minvalz, maxvalz])
    ax.set_ylim3d([minvaly, maxvaly])"""

new_limits = """    if bounds:
        ax.set_xlim3d([bounds['x_min'], bounds['x_max']])
        ax.set_ylim3d([bounds['y_min'], bounds['y_max']])
        ax.set_zlim3d([bounds['z_min'], bounds['z_max']])
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
    else:
        minvaly, maxvaly = np.min(joints3d_25_3[:, 2]), np.max(joints3d_25_3[:, 2])
        minvalz, maxvalz = np.min(joints3d_25_3[:, 1]), np.max(joints3d_25_3[:, 1])
        ax.set_xlim3d([-1.0, 2.0])
        ax.set_zlim3d([minvalz, maxvalz])
        ax.set_ylim3d([minvaly, maxvaly])"""

content = content.replace(old_limits, new_limits)

old_render_call = """    for i in range(t):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i], render_width, render_height, "GT 3D Pose"), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i], render_width, render_height, "Pred 3D Pose"), cv2.COLOR_RGB2BGR)"""

new_render_call = """    bounds = _compute_3d_bounds(gt_joints_t_25_3, pred_joints_t_25_3)
    for i in range(t):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i], render_width, render_height, "Pred 3D Pose", bounds), cv2.COLOR_RGB2BGR)"""

content = content.replace(old_render_call, new_render_call)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

