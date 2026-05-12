import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# I see, my previous patch didn't successfully replace _render_sequence_cells_emg2pose.
# Let's just redefine it completely.
render_emg2pose_new = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    
    # Update bounds to include our prediction
    all_j = [gt_joints_t_25_3[:, :25, :], pred_joints_t_25_3[:, :25, :]]
    if our_pred_joints_t_25_3 is not None:
        all_j.append(our_pred_joints_t_25_3[:, :25, :])
    all_joints = np.concatenate(all_j, axis=0)
    x = all_joints[..., 0]
    y = all_joints[..., 2]
    z = all_joints[..., 1]
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0 * 1.1
    bounds = {
        'x_min': x_mid - max_range, 'x_max': x_mid + max_range,
        'y_min': y_mid - max_range, 'y_max': y_mid + max_range,
        'z_min': z_mid - max_range, 'z_max': z_mid + max_range,
    }

    frames = []
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        
        rows = [
            np.concatenate([gt_skel, panel_emg], axis=1),
            np.concatenate([pred_skel, panel_emg], axis=1)
        ]
        
        if our_pred_joints_t_25_3 is not None:
            our_skel = cv2.cvtColor(_render_skeleton_panel(our_pred_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
            rows.append(np.concatenate([our_skel, panel_emg], axis=1))
            
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Official:Red)", bounds), cv2.COLOR_RGB2BGR)
        rows.append(np.concatenate([overlay_skel, panel_emg], axis=1))
        
        frames.append(np.concatenate(rows, axis=0))
    return frames"""

# Find the def and replace until the next def
import re
content = re.sub(r'def _render_sequence_cells_emg2pose\(.*?return frames', render_emg2pose_new, content, flags=re.DOTALL)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
