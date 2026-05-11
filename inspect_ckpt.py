import torch
ckpt = torch.load('/data/litengmo/HSMR/mia_custom/custom/tools/official_eval/output/20260510_000126/checkpoints/official_reproduction_cond_emgtopose_threed/model_100.pth', map_location='cpu', weights_only=False)
print("Model args in ckpt:", ckpt.get('model_args'))
print("Train args in ckpt:", ckpt.get('train_args').predemg)
print("conv1.weight shape:", ckpt['my_model']['conv1.weight'].shape)
