import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Update emg2pose signature to have 11 args
old_sig = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:"""

new_sig = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:"""

# Actually, the error is: "takes 10 positional arguments but 11 were given"
# Wait, old_sig has 11 args: gt, pred, our_pred, emg, fps, plot_w, plot_h, render_w, render_h, vmax, debug
# Let's check how it's defined in the file
import inspect
import sys
sys.path.insert(0, '/data/litengmo/HSMR/mia_custom/custom/vis')
import vis_infer_final
print(inspect.signature(vis_infer_final._render_sequence_cells_emg2pose))

