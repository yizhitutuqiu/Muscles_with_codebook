import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

old_func = """def _infer_our_emg2pose(model, emg_8_t, condval, device, joints3d_t_25_3) -> np.ndarray:
    import torch
    inputs_np = emg_8_t.T
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(inputs, cond=cond)
        pred = out["pred"]
    pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # Restore absolute coordinates using GT's Joint 8 (which is the root in Mia_style_eval)
    joints3d_t_25_3 = joints3d_t_25_3[:, :25, :]
    root_8 = joints3d_t_25_3[:, 8:9, :]
    return pred_np + root_8"""

new_func = """def _infer_our_emg2pose(model, stage1, emg_8_t, condval, device, joints3d_t_25_3) -> np.ndarray:
    import torch
    inputs_np = emg_8_t.T
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device) # B, T, 8
    
    # STANDARDIZE EMG
    stage1_emg = getattr(stage1, "emg", None)
    emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
    if emg_standardizer is not None:
        from custom.tools.Mia_style_eval import _emg_standardizer_stats_bt8
        mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(inputs.shape[1]), device=device)
        inputs = (inputs - mean_bt8) / std_bt8
        
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(inputs, cond=cond)
        pred = out["pred"]
    pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    # Restore absolute coordinates using GT's Joint 8 (which is the root in Mia_style_eval)
    joints3d_t_25_3 = joints3d_t_25_3[:, :25, :]
    root_8 = joints3d_t_25_3[:, 8:9, :]
    return pred_np + root_8"""

content = content.replace(old_func, new_func)

# Also update the call to _infer_our_emg2pose
call_old = 'our_pred_joints = _infer_our_emg2pose(our_model, item["raw_emg"], condval, device, item["raw_joints"])'
call_new = 'our_pred_joints = _infer_our_emg2pose(our_model, our_stage1, item["raw_emg"], condval, device, item["raw_joints"])'
content = content.replace(call_old, call_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

