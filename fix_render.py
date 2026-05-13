import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Replace the inner part of _render_sequence_cells_emg2pose using regex
pattern = re.compile(
    r'    bounds = _compute_3d_bounds\(gt_joints_t_25_3, pred_joints_t_25_3\)\n'
    r'    for i in tqdm\(range\(t\), desc="Rendering emg2pose frames", leave=False\):\n'
    r'.*?return frames',
    re.DOTALL
)

new_body = """    bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3, pred_joints_t_25_3, our_joints_t_25_3], axis=0), np.zeros_like(gt_joints_t_25_3)) # HACK: compute_3d_bounds handles concatenated
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        our_skel = cv2.cvtColor(_render_skeleton_panel(our_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], our_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Our:Red)", bounds), cv2.COLOR_RGB2BGR)
        
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        our_row = np.concatenate([our_skel, panel_emg], axis=1)
        overlay_row = np.concatenate([overlay_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row, our_row, overlay_row], axis=0))
    return frames"""

content = pattern.sub(new_body, content)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
