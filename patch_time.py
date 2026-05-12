import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

if 'import time' not in content:
    content = content.replace('import sys', 'import sys\nimport time')

# Add counters
content = content.replace(
    '    for exercise, samples_in_ex in tqdm',
    '    total_inference_time = 0.0\n    total_visualization_time = 0.0\n    for exercise, samples_in_ex in tqdm'
)

# Wrap pose2emg inference
content = content.replace(
    '            if task == "pose2emg":\n                pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)',
    '            if task == "pose2emg":\n                t0 = time.time()\n                pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)\n                total_inference_time += time.time() - t0'
)

# Wrap emg2pose inference
content = content.replace(
    '            else:\n                pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)',
    '            else:\n                t0 = time.time()\n                pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)\n                total_inference_time += time.time() - t0'
)

# Wrap visualization start
content = content.replace(
    '        cell_streams = []\n        for item in prepared:',
    '        t1 = time.time()\n        cell_streams = []\n        for item in prepared:'
)

# Wrap visualization end
content = content.replace(
    '        finally:\n            writer.release()',
    '        finally:\n            writer.release()\n        total_visualization_time += time.time() - t1'
)

# Add print at the end
content = content.replace(
    'if __name__ == "__main__":',
    '    print(f"\\n--- Time Statistics ---")\n    print(f"Total Inference Time: {total_inference_time:.2f} s")\n    print(f"Total Visualization Time: {total_visualization_time:.2f} s")\n    if total_inference_time > 0:\n        print(f"Visualization takes {total_visualization_time / total_inference_time:.1f}x longer than inference.")\n\nif __name__ == "__main__":'
)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

