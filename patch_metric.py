import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

metric_call_old = 'metric = _compute_official_metrics(pred_joints, arrays["joints3d_t_25_3"], task)'
metric_call_new = 'metric = _compute_official_metrics(pred_joints, arrays["joints3d_t_25_3"][:, :25, :], task)'

content = content.replace(metric_call_old, metric_call_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
