import torch
import sys

def analyze_dstformer(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = state.get('model_state', state)
    
    temporal_keys = [k for k in state_dict.keys() if k.startswith('temporal.')]
    
    total_dst_params = sum(state_dict[k].numel() for k in temporal_keys)
    
    # Calculate MoE vs Attn params
    moe_params = sum(state_dict[k].numel() for k in temporal_keys if 'experts' in k or 'router' in k)
    attn_params = sum(state_dict[k].numel() for k in temporal_keys if 'attn' in k and 'experts' not in k)
    
    # Get number of blocks
    blocks = set([k.split('.')[3] for k in temporal_keys if len(k.split('.')) > 3 and k.split('.')[2] == 'blocks'])
    num_blocks = len(blocks)
    
    print(f"Total Temporal (DSTFormer MoE) params: {total_dst_params:,}")
    print(f"Number of Blocks: {num_blocks}")
    print(f"MoE FFN Params: {moe_params:,} ({moe_params/total_dst_params*100:.1f}%)")
    print(f"Attention Params: {attn_params:,} ({attn_params/total_dst_params*100:.1f}%)")

analyze_dstformer("/data/litengmo/HSMR/mia_custom/custom/ablations/batch_ablation_prior/exp_1_pure_continuous/best.pt")
