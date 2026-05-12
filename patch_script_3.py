import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

overlay_func = """def _render_overlay_skeleton_panel(gt_joints_25_3: np.ndarray, pred_joints_25_3: np.ndarray, width: int, height: int, title: str, bounds: dict = None) -> np.ndarray:
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

    if bounds:
        ax.set_xlim3d([bounds['x_min'], bounds['x_max']])
        ax.set_ylim3d([bounds['y_min'], bounds['y_max']])
        ax.set_zlim3d([bounds['z_min'], bounds['z_max']])
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
    else:
        minvaly = min(np.min(gt_joints_25_3[:, 2]), np.min(pred_joints_25_3[:, 2]))
        maxvaly = max(np.max(gt_joints_25_3[:, 2]), np.max(pred_joints_25_3[:, 2]))
        minvalz = min(np.min(gt_joints_25_3[:, 1]), np.min(pred_joints_25_3[:, 1]))
        maxvalz = max(np.max(gt_joints_25_3[:, 1]), np.max(pred_joints_25_3[:, 1]))
        ax.set_xlim3d([-1.0, 2.0])
        ax.set_zlim3d([minvalz, maxvalz])
        ax.set_ylim3d([minvaly, maxvaly])

    ax.invert_zaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(0, 180)

    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"): img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    else:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def _resize_letterbox"""

content = content.replace("def _resize_letterbox", overlay_func)

render_call_old = """        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row], axis=0))"""

render_call_new = """        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Pred:Red)", bounds), cv2.COLOR_RGB2BGR)
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        overlay_row = np.concatenate([overlay_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row, overlay_row], axis=0))"""

content = content.replace(render_call_old, render_call_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

