import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

patch_code = """
                item["our_pred_joints"] = _pad_time_first(item["our_pred_joints"], max_t)
                
                print(f"Shapes: GT {item['gt_joints'].shape}, Pred {item['pred_joints'].shape}, Our {item['our_pred_joints'].shape}")
                
                cell_frames = _render_sequence_cells_emg2pose(
"""
content = content.replace(
    '                item["our_pred_joints"] = _pad_time_first(item["our_pred_joints"], max_t)\n                cell_frames = _render_sequence_cells_emg2pose(',
    patch_code
)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
