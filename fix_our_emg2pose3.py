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
    return pred_np"""

new_func = """def _infer_our_emg2pose(model, emg_8_t, condval, device, joints3d_t_25_3) -> np.ndarray:
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

content = content.replace(old_func, new_func)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

