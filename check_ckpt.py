import sys, torch
import glob
for f in glob.glob("/data/litengmo/HSMR/mia_custom/pretrained-checkpoints/*/*.pth"):
    try:
        c = torch.load(f, map_location='cpu', weights_only=False)
        shape = c.get('my_model', {}).get('conv1.weight', torch.empty(0)).shape
        print(f"{f}: {shape}")
    except Exception as e:
        print(f"{f}: {e}")
