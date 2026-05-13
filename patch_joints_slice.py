import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

content = content.replace(
    'root = joints3d_t_25_3[:, 0:1, :]',
    'joints3d_t_25_3 = joints3d_t_25_3[:, :25, :]\n    root = joints3d_t_25_3[:, 0:1, :]'
)

content = content.replace(
    'pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)\n    root = joints3d_t_25_3[:, 0:1, :]',
    'pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)\n    joints3d_t_25_3 = joints3d_t_25_3[:, :25, :]\n    root = joints3d_t_25_3[:, 0:1, :]'
)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
