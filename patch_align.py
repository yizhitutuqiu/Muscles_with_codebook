import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# 1. Update _render_overlay_skeleton_panel
old_overlay = """def _render_overlay_skeleton_panel(gt_joints_25_3: np.ndarray, pred_joints_25_3: np.ndarray, width: int, height: int, title: str, bounds: dict = None) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    
    limb_seq = [([17, 15], None), ([15, 0], None), ([0, 16], None), ([16, 18], None), ([0, 1], None), ([1, 2], None), ([2, 3], None), ([3, 4], None), ([1, 5], None), ([5, 6], None), ([6, 7], None), ([1, 8], None), ([8, 9], None), ([9, 10], None), ([10, 24], None), ([8, 12], None), ([12, 13], None), ([13, 14], None), ([24, 22], None), ([24, 24], None), ([22, 23], None), ([14, 19], None), ([14, 21], None), ([19, 20], None)]
    
    # GT (Green)
    for j in range(25):
        ax.scatter3D(gt_joints_25_3[j, 0], gt_joints_25_3[j, 2], gt_joints_25_3[j, 1], c='green', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([gt_joints_25_3[vertices[0], 0], gt_joints_25_3[vertices[1], 0]],
                  [gt_joints_25_3[vertices[0], 2], gt_joints_25_3[vertices[1], 2]],
                  [gt_joints_25_3[vertices[0], 1], gt_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='green', alpha=0.7)
                  
    # Pred (Red)
    for j in range(25):
        ax.scatter3D(pred_joints_25_3[j, 0], pred_joints_25_3[j, 2], pred_joints_25_3[j, 1], c='red', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([pred_joints_25_3[vertices[0], 0], pred_joints_25_3[vertices[1], 0]],
                  [pred_joints_25_3[vertices[0], 2], pred_joints_25_3[vertices[1], 2]],
                  [pred_joints_25_3[vertices[0], 1], pred_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='red', alpha=0.7)

    if bounds:"""

new_overlay = """def _render_overlay_skeleton_panel(gt_joints_25_3: np.ndarray, pred_joints_25_3: np.ndarray, our_joints_25_3: np.ndarray, width: int, height: int, title: str, bounds: dict = None) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title(title)
    
    limb_seq = [([17, 15], None), ([15, 0], None), ([0, 16], None), ([16, 18], None), ([0, 1], None), ([1, 2], None), ([2, 3], None), ([3, 4], None), ([1, 5], None), ([5, 6], None), ([6, 7], None), ([1, 8], None), ([8, 9], None), ([9, 10], None), ([10, 24], None), ([8, 12], None), ([12, 13], None), ([13, 14], None), ([24, 22], None), ([24, 24], None), ([22, 23], None), ([14, 19], None), ([14, 21], None), ([19, 20], None)]
    
    # GT (Green)
    for j in range(25):
        ax.scatter3D(gt_joints_25_3[j, 0], gt_joints_25_3[j, 2], gt_joints_25_3[j, 1], c='green', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([gt_joints_25_3[vertices[0], 0], gt_joints_25_3[vertices[1], 0]],
                  [gt_joints_25_3[vertices[0], 2], gt_joints_25_3[vertices[1], 2]],
                  [gt_joints_25_3[vertices[0], 1], gt_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='green', alpha=0.7)
                  
    # Official Pred (Red)
    for j in range(25):
        ax.scatter3D(pred_joints_25_3[j, 0], pred_joints_25_3[j, 2], pred_joints_25_3[j, 1], c='red', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([pred_joints_25_3[vertices[0], 0], pred_joints_25_3[vertices[1], 0]],
                  [pred_joints_25_3[vertices[0], 2], pred_joints_25_3[vertices[1], 2]],
                  [pred_joints_25_3[vertices[0], 1], pred_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='red', alpha=0.7)
                  
    # Our Pred (Blue)
    for j in range(25):
        ax.scatter3D(our_joints_25_3[j, 0], our_joints_25_3[j, 2], our_joints_25_3[j, 1], c='blue', s=15)
    for vertices, _ in limb_seq:
        ax.plot3D([our_joints_25_3[vertices[0], 0], our_joints_25_3[vertices[1], 0]],
                  [our_joints_25_3[vertices[0], 2], our_joints_25_3[vertices[1], 2]],
                  [our_joints_25_3[vertices[0], 1], our_joints_25_3[vertices[1], 1]],
                  linewidth=2, color='blue', alpha=0.7)

    if bounds:"""

content = content.replace(old_overlay, new_overlay)

# Update limits calculation in _render_overlay_skeleton_panel
old_limits = """        minvaly = min(np.min(gt_joints_25_3[:, 2]), np.min(pred_joints_25_3[:, 2]))
        maxvaly = max(np.max(gt_joints_25_3[:, 2]), np.max(pred_joints_25_3[:, 2]))
        minvalz = min(np.min(gt_joints_25_3[:, 1]), np.min(pred_joints_25_3[:, 1]))
        maxvalz = max(np.max(gt_joints_25_3[:, 1]), np.max(pred_joints_25_3[:, 1]))"""

new_limits = """        minvaly = min(np.min(gt_joints_25_3[:, 2]), np.min(pred_joints_25_3[:, 2]), np.min(our_joints_25_3[:, 2]))
        maxvaly = max(np.max(gt_joints_25_3[:, 2]), np.max(pred_joints_25_3[:, 2]), np.max(our_joints_25_3[:, 2]))
        minvalz = min(np.min(gt_joints_25_3[:, 1]), np.min(pred_joints_25_3[:, 1]), np.min(our_joints_25_3[:, 1]))
        maxvalz = max(np.max(gt_joints_25_3[:, 1]), np.max(pred_joints_25_3[:, 1]), np.max(our_joints_25_3[:, 1]))"""

content = content.replace(old_limits, new_limits)


# 2. Update _render_sequence_cells_emg2pose
old_e2p_sig = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3[:, :25], pred_joints_t_25_3[:, :25], our_joints_t_25_3[:, :25]], axis=0), np.zeros_like(gt_joints_t_25_3[:, :25])) # HACK: compute_3d_bounds handles concatenated
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        our_skel = cv2.cvtColor(_render_skeleton_panel(our_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], our_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Our:Red)", bounds), cv2.COLOR_RGB2BGR)"""

new_e2p_sig = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool,
    align_root_to_gt: bool = False
) -> list[np.ndarray]:
    if align_root_to_gt:
        pred_joints_t_25_3 = pred_joints_t_25_3 - pred_joints_t_25_3[:, 0:1, :] + gt_joints_t_25_3[:, 0:1, :]
        our_joints_t_25_3 = our_joints_t_25_3 - our_joints_t_25_3[:, 0:1, :] + gt_joints_t_25_3[:, 0:1, :]

    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3[:, :25], pred_joints_t_25_3[:, :25], our_joints_t_25_3[:, :25]], axis=0), np.zeros_like(gt_joints_t_25_3[:, :25])) # HACK: compute_3d_bounds handles concatenated
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        our_skel = cv2.cvtColor(_render_skeleton_panel(our_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], our_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:G, Off:R, Ours:B)", bounds), cv2.COLOR_RGB2BGR)"""

content = content.replace(old_e2p_sig, new_e2p_sig)

# 3. Add align_root_to_gt to config parsing
cfg_parse_old = """    debug_color_stats = cfg.get("debug_color_stats", False)
    debug_overlay_text = cfg.get("debug_overlay_text", False)
    dry_run = cfg.get("dry_run", False)"""

cfg_parse_new = """    debug_color_stats = cfg.get("debug_color_stats", False)
    debug_overlay_text = cfg.get("debug_overlay_text", False)
    dry_run = cfg.get("dry_run", False)
    align_root_to_gt = cfg.get("align_root_to_gt", False)"""

content = content.replace(cfg_parse_old, cfg_parse_new)

# 4. Pass align_root_to_gt in main
call_e2p_old = """                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["our_pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )"""

call_e2p_new = """                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["our_pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text, align_root_to_gt
                )"""

content = content.replace(call_e2p_old, call_e2p_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
