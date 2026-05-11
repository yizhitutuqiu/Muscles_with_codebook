import torch
ckpt = torch.load("/data/litengmo/HSMR/mia_custom/custom/checkpoints/new_clip5_codebook_with_exerciseloss/exp_shared_head/best.pt", map_location='cpu')
if 'config' in ckpt:
    import yaml
    print(yaml.dump(ckpt['config'], default_flow_style=False))
else:
    print("No config found in checkpoint")
