import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Fix bounds computation inside _render_sequence_cells_emg2pose
content = content.replace(
    'bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3, pred_joints_t_25_3, our_joints_t_25_3], axis=0), np.zeros_like(gt_joints_t_25_3))',
    'bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3[:, :25], pred_joints_t_25_3[:, :25], our_joints_t_25_3[:, :25]], axis=0), np.zeros_like(gt_joints_t_25_3[:, :25]))'
)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
