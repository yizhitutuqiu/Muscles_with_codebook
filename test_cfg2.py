import torch
ckpt = torch.load('/data/litengmo/HSMR/mia_custom/custom/stage2/checkpoints/batch_emg2pose_ablation/exp_emg2pose_ablation_h8_baseline/best.pt', map_location='cpu')
cfg = ckpt.get('config', {})
print("joints3d_root_center:", cfg.get('data', {}).get('joints3d_root_center', 'NOT_FOUND'))
