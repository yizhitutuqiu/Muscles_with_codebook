import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

content = content.replace('print(f"Shapes: GT {item[\'gt_joints\'].shape}, Pred {item[\'pred_joints\'].shape}, Our {item[\'our_pred_joints\'].shape}")', '')

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
