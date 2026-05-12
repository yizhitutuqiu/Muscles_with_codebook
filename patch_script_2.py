import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Fix _compute_3d_bounds
bounds_code_old = """def _compute_3d_bounds(gt_joints: np.ndarray, pred_joints: np.ndarray) -> dict:
    all_joints = np.concatenate([gt_joints, pred_joints], axis=0)"""
bounds_code_new = """def _compute_3d_bounds(gt_joints: np.ndarray, pred_joints: np.ndarray) -> dict:
    gt_j = gt_joints[:, :25, :]
    pred_j = pred_joints[:, :25, :]
    all_joints = np.concatenate([gt_j, pred_j], axis=0)"""
content = content.replace(bounds_code_old, bounds_code_new)

# Fix _render_sequence_cells_emg2pose slicing
render_call_old = """    bounds = _compute_3d_bounds(gt_joints_t_25_3, pred_joints_t_25_3)
    for i in range(t):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i], render_width, render_height, "Pred 3D Pose", bounds), cv2.COLOR_RGB2BGR)"""

render_call_new = """    bounds = _compute_3d_bounds(gt_joints_t_25_3, pred_joints_t_25_3)
    for i in range(t):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Pred 3D Pose", bounds), cv2.COLOR_RGB2BGR)"""

content = content.replace(render_call_old, render_call_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

