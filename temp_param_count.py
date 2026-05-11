import torch
import sys

def analyze_params(ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Determine where the actual model state dict is
        if 'model_state' in state:
            state_dict = state['model_state']
        elif 'my_model' in state:
            state_dict = state['my_model']
        elif 'model_state_dict' in state:
            state_dict = state['model_state_dict']
        elif 'model' in state and isinstance(state['model'], dict):
            state_dict = state['model']
        else:
            state_dict = state
            
        total_params = 0
        module_params = {}
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                num = param.numel()
                total_params += num
                
                # Get top-level module name
                top_level = name.split('.')[0]
                if top_level not in module_params:
                    module_params[top_level] = 0
                module_params[top_level] += num
                
        return total_params, module_params
    except Exception as e:
        return str(e), {}

paths = {
    "exp_1_pure_continuous": "/data/litengmo/HSMR/mia_custom/custom/ablations/batch_ablation_prior/exp_1_pure_continuous/best.pt",
    "official_cond_posetoemg": "/data/litengmo/HSMR/mia_custom/pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth"
}

for name, path in paths.items():
    total, mod_params = analyze_params(path)
    if isinstance(total, int):
        print(f"[{name}]")
        print(f"Total Parameters: {total:,}")
        print("Module breakdown:")
        # Sort modules by size descending
        sorted_mods = sorted(mod_params.items(), key=lambda x: x[1], reverse=True)
        for mod, num in sorted_mods:
            print(f"  - {mod}: {num:,} ({num/total*100:.1f}%)")
    else:
        print(f"[{name}]: Error loading -> {total}")
    print("-" * 70)
