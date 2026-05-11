import torch
ckpt = torch.load('/data/litengmo/HSMR/mia_custom/custom/checkpoints/new_clip5_codebook_with_exerciseloss/exp_shared_head/best.pt', map_location='cpu')
print(ckpt['config']['model']['vq']['num_codes'])
