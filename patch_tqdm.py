import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Add import
if 'from tqdm import tqdm' not in content:
    content = content.replace('import yaml', 'import yaml\nfrom tqdm import tqdm')

# Replace the main loop
content = content.replace(
    'for exercise, samples_in_ex in sorted(selected_by_exercise.items(), key=lambda x: x[0]):',
    'for exercise, samples_in_ex in tqdm(sorted(selected_by_exercise.items(), key=lambda x: x[0]), desc="Processing exercises"):'
)

# Replace the frame loop in _render_sequence_cells_pose2emg
content = content.replace(
    '    for i in range(t):\n        verts, cam',
    '    for i in tqdm(range(t), desc="Rendering pose2emg frames", leave=False):\n        verts, cam'
)

# Replace the frame loop in _render_sequence_cells_emg2pose
content = content.replace(
    '    for i in range(t):\n        gt_skel',
    '    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):\n        gt_skel'
)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

