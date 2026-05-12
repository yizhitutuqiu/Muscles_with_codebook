import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Add config parsing for our models
cfg_load_old = """    if task == "pose2emg":
        checkpoint_path = Path(cfg.get("pose2emg", {}).get("checkpoint", ""))
    elif task == "emg2pose":
        checkpoint_path = Path(cfg.get("emg2pose", {}).get("checkpoint", ""))"""

cfg_load_new = """    if task == "pose2emg":
        checkpoint_path = Path(cfg.get("pose2emg", {}).get("checkpoint", ""))
        our_checkpoint_path = Path(cfg.get("pose2emg", {}).get("our_checkpoint", ""))
    elif task == "emg2pose":
        checkpoint_path = Path(cfg.get("emg2pose", {}).get("checkpoint", ""))
        our_checkpoint_path = Path(cfg.get("emg2pose", {}).get("our_checkpoint", ""))"""

content = content.replace(cfg_load_old, cfg_load_new)

# Modify model loading to support our model
model_load_old = """    model = _load_model(checkpoint_path, device, task)"""

model_load_new = """    model = _load_model(checkpoint_path, device, task)
    from custom.tools.Mia_style_eval import _build_stage2_from_ckpt
    try:
        our_model, _, _, _ = _build_stage2_from_ckpt(our_checkpoint_path, device, stage1_override_ckpt=None)
        our_model.eval()
        has_our_model = True
    except Exception as e:
        print(f"Warning: Could not load our model from {our_checkpoint_path}. Error: {e}")
        has_our_model = False"""

content = content.replace(model_load_old, model_load_new)

# Add our inference functions
infer_funcs = """def _infer_our_emg(model: Any, joints3d_t_25_3: np.ndarray, condval: float, device: Any) -> np.ndarray:
    import torch
    t = joints3d_t_25_3.shape[0]
    # our model expects (B, T, V, 3) for joints3d
    x = torch.from_numpy(joints3d_t_25_3).unsqueeze(0).to(device)
    cond = torch.tensor([condval], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(joints3d=x, cond=cond)
        pred = out["emg"] # (B, T, 8)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

def _infer_our_pose(model: Any, emg_8_t: np.ndarray, condval: float, device: Any) -> np.ndarray:
    import torch
    # our model expects (B, T, 8) for emg
    x = torch.from_numpy(emg_8_t.T).unsqueeze(0).to(device)
    cond = torch.tensor([condval], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(emg=x, cond=cond)
        pred = out["joints3d"] # (B, T, V, 3)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
"""

content = content.replace('def _infer_emg', infer_funcs + '\ndef _infer_emg')

# Update caching logic to also cache our model's predictions
cache_logic_old = """            cache_file = temp_dir / f"{sample.sample_id}_pred.npy"
            metric_file = temp_dir / f"{sample.sample_id}_metric.json"
            
            if task == "pose2emg":
                if cache_file.exists():
                    pred_emg_8_t = np.load(cache_file)
                else:
                    t0 = time.time()
                    pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)
                    total_inference_time += time.time() - t0
                    np.save(cache_file, pred_emg_8_t)
                
                if not metric_file.exists():
                    gt_flat = arrays["emg_8_t"].T # t, 8
                    metric = _compute_official_metrics(pred_emg_8_t.T, gt_flat, task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                gt_emg_mesh = _normalize_emg_for_mesh(arrays["emg_8_t"], sample.subject)
                pred_emg_mesh = _normalize_emg_for_mesh(pred_emg_8_t, sample.subject)
                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh,
                })
            else:
                if cache_file.exists():
                    pred_joints = np.load(cache_file)
                else:
                    t0 = time.time()
                    pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)
                    total_inference_time += time.time() - t0
                    np.save(cache_file, pred_joints)
                
                if not metric_file.exists():
                    metric = _compute_official_metrics(pred_joints, arrays["joints3d_t_25_3"][:, :25, :], task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32)
                })"""

cache_logic_new = """            cache_file = temp_dir / f"{sample.sample_id}_pred.npy"
            our_cache_file = temp_dir / f"{sample.sample_id}_our_pred.npy"
            metric_file = temp_dir / f"{sample.sample_id}_metric.json"
            
            if task == "pose2emg":
                if cache_file.exists():
                    pred_emg_8_t = np.load(cache_file)
                else:
                    t0 = time.time()
                    pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)
                    total_inference_time += time.time() - t0
                    np.save(cache_file, pred_emg_8_t)
                    
                our_pred_emg_8_t = None
                if has_our_model:
                    if our_cache_file.exists():
                        our_pred_emg_8_t = np.load(our_cache_file)
                    else:
                        # only run our model inference if we are going to use it
                        # but actually we are in phase 1, so run it
                        t0 = time.time()
                        our_pred_emg_8_t = _infer_our_emg(our_model, arrays["joints3d_t_25_3"], condval, device)
                        our_pred_emg_8_t = our_pred_emg_8_t.T # (8, T)
                        total_inference_time += time.time() - t0
                        np.save(our_cache_file, our_pred_emg_8_t)
                
                if not metric_file.exists():
                    gt_flat = arrays["emg_8_t"].T # t, 8
                    metric = _compute_official_metrics(pred_emg_8_t.T, gt_flat, task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                gt_emg_mesh = _normalize_emg_for_mesh(arrays["emg_8_t"], sample.subject)
                pred_emg_mesh = _normalize_emg_for_mesh(pred_emg_8_t, sample.subject)
                our_pred_emg_mesh = _normalize_emg_for_mesh(our_pred_emg_8_t, sample.subject) if our_pred_emg_8_t is not None else None
                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "our_pred_emg_plot": our_pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh, "our_pred_emg_mesh": our_pred_emg_mesh
                })
            else:
                if cache_file.exists():
                    pred_joints = np.load(cache_file)
                else:
                    t0 = time.time()
                    pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)
                    total_inference_time += time.time() - t0
                    np.save(cache_file, pred_joints)
                    
                our_pred_joints = None
                if has_our_model:
                    if our_cache_file.exists():
                        our_pred_joints = np.load(our_cache_file)
                    else:
                        t0 = time.time()
                        our_pred_joints = _infer_our_pose(our_model, arrays["emg_8_t"], condval, device)
                        total_inference_time += time.time() - t0
                        np.save(our_cache_file, our_pred_joints)
                
                if not metric_file.exists():
                    metric = _compute_official_metrics(pred_joints, arrays["joints3d_t_25_3"][:, :25, :], task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "our_pred_joints": our_pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32)
                })"""
content = content.replace(cache_logic_old, cache_logic_new)

# Update visualization logic to include our model
# We need to add the third row for our model
render_pose2emg_old = """def _render_sequence_cells_pose2emg(
    renderer: Any, background_bgr: np.ndarray, verts_t_v_3: np.ndarray, origcam_t_4: np.ndarray,
    gt_emg_mesh_8_t: np.ndarray, pred_emg_mesh_8_t: np.ndarray, gt_emg_plot_8_t: np.ndarray, pred_emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, plot_vmax: float, mesh_views: str, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_emg_mesh_8_t.shape[1]
    panel_gt = cv2.cvtColor(_render_emg_panel(gt_emg_plot_8_t, plot_width, plot_height, "GT EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(_render_emg_panel(pred_emg_plot_8_t, plot_width, plot_height, "Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    for i in tqdm(range(t), desc="Rendering pose2emg frames", leave=False):
        verts, cam = verts_t_v_3[i], origcam_t_4[i]
        meshes = []
        for is_pred, emg_mesh in [(False, gt_emg_mesh_8_t), (True, pred_emg_mesh_8_t)]:
            views = []
            if mesh_views in ["front", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=True, pred=is_pred)
                views.append(v)
            if mesh_views in ["back", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=False, pred=is_pred)
                views.append(v)
            meshes.append(np.concatenate(views, axis=1))
        
        gt_mesh, pred_mesh = meshes
        frames.append(np.concatenate([np.concatenate([gt_mesh, panel_gt], axis=1), np.concatenate([pred_mesh, panel_pred], axis=1)], axis=0))
    return frames"""

render_pose2emg_new = """def _render_sequence_cells_pose2emg(
    renderer: Any, background_bgr: np.ndarray, verts_t_v_3: np.ndarray, origcam_t_4: np.ndarray,
    gt_emg_mesh_8_t: np.ndarray, pred_emg_mesh_8_t: np.ndarray, gt_emg_plot_8_t: np.ndarray, pred_emg_plot_8_t: np.ndarray,
    our_pred_emg_mesh_8_t: np.ndarray, our_pred_emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, plot_vmax: float, mesh_views: str, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_emg_mesh_8_t.shape[1]
    panel_gt = cv2.cvtColor(_render_emg_panel(gt_emg_plot_8_t, plot_width, plot_height, "GT EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(_render_emg_panel(pred_emg_plot_8_t, plot_width, plot_height, "Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    if our_pred_emg_plot_8_t is not None:
        panel_our = cv2.cvtColor(_render_emg_panel(our_pred_emg_plot_8_t, plot_width, plot_height, "Our Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
        
    frames = []
    for i in tqdm(range(t), desc="Rendering pose2emg frames", leave=False):
        verts, cam = verts_t_v_3[i], origcam_t_4[i]
        meshes = []
        render_list = [(False, gt_emg_mesh_8_t), (True, pred_emg_mesh_8_t)]
        if our_pred_emg_mesh_8_t is not None:
            render_list.append((True, our_pred_emg_mesh_8_t))
            
        for is_pred, emg_mesh in render_list:
            views = []
            if mesh_views in ["front", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=True, pred=is_pred)
                views.append(v)
            if mesh_views in ["back", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=False, pred=is_pred)
                views.append(v)
            meshes.append(np.concatenate(views, axis=1))
        
        gt_row = np.concatenate([meshes[0], panel_gt], axis=1)
        pred_row = np.concatenate([meshes[1], panel_pred], axis=1)
        rows = [gt_row, pred_row]
        if our_pred_emg_mesh_8_t is not None:
            our_row = np.concatenate([meshes[2], panel_our], axis=1)
            rows.insert(1, our_row) # Put it before Pred, or after Pred? User said "final visualization add a row at the penultimate row" (倒数第二行加一行) -> So order: GT, Our, Pred? Wait, user said "最下面一行是叠加". So rows were: GT, Pred, Overlay. Penultimate means before Overlay. So order: GT, Pred, Our, Overlay? Wait, pose2emg has no overlay!
            # Let's just put GT, Our, Pred. User said "在倒数第二行加一行", which means if previously there were 3 rows (GT, Pred, Overlay), it becomes GT, Pred, Our, Overlay.
            # But for pose2emg, previously there were 2 rows. "倒数第二行" would mean between GT and Pred? Or just add it. Let's make it: GT, Our, Pred.
            # Or GT, Pred, Our. I'll make it GT, Our, Pred.
            # Wait, user said "倒数第二行加一行" - meaning the last row is Overlay.
            # So: GT, Pred, Our, Overlay. Let's just append to rows if no overlay.
            pass
        frames.append(np.concatenate(rows, axis=0))
    return frames"""
content = content.replace(render_pose2emg_old, render_pose2emg_new)

# Update emg2pose render function
render_emg2pose_old = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    bounds = _compute_3d_bounds(gt_joints_t_25_3, pred_joints_t_25_3)
    frames = []
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Pred 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Pred:Red)", bounds), cv2.COLOR_RGB2BGR)
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        overlay_row = np.concatenate([overlay_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row, overlay_row], axis=0))
    return frames"""

render_emg2pose_new = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:
    t = gt_joints_t_25_3.shape[0]
    panel_emg = cv2.cvtColor(_render_emg_panel(emg_plot_8_t, plot_width, plot_height, "Input EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    
    # Update bounds to include our prediction
    all_j = [gt_joints_t_25_3[:, :25, :], pred_joints_t_25_3[:, :25, :]]
    if our_pred_joints_t_25_3 is not None:
        all_j.append(our_pred_joints_t_25_3[:, :25, :])
    all_joints = np.concatenate(all_j, axis=0)
    x = all_joints[..., 0]
    y = all_joints[..., 2]
    z = all_joints[..., 1]
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    z_mid = (z.max() + z.min()) / 2
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0 * 1.1
    bounds = {
        'x_min': x_mid - max_range, 'x_max': x_mid + max_range,
        'y_min': y_mid - max_range, 'y_max': y_mid + max_range,
        'z_min': z_mid - max_range, 'z_max': z_mid + max_range,
    }

    frames = []
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        
        rows = [
            np.concatenate([gt_skel, panel_emg], axis=1),
            np.concatenate([pred_skel, panel_emg], axis=1)
        ]
        
        if our_pred_joints_t_25_3 is not None:
            our_skel = cv2.cvtColor(_render_skeleton_panel(our_pred_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
            rows.append(np.concatenate([our_skel, panel_emg], axis=1))
            
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Official:Red)", bounds), cv2.COLOR_RGB2BGR)
        rows.append(np.concatenate([overlay_skel, panel_emg], axis=1))
        
        frames.append(np.concatenate(rows, axis=0))
    return frames"""

content = content.replace(render_emg2pose_old, render_emg2pose_new)

# Update Phase 2 call mapping
phase2_call_pose2emg_old = """                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )"""
phase2_call_pose2emg_new = """                if item.get("our_pred_emg_plot") is not None:
                    item["our_pred_emg_plot"] = _pad_emg_8_t(item["our_pred_emg_plot"], max_t)
                    item["our_pred_emg_mesh"] = _pad_emg_8_t(item["our_pred_emg_mesh"], max_t)
                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], item.get("our_pred_emg_mesh"), item.get("our_pred_emg_plot"),
                    fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )"""
content = content.replace(phase2_call_pose2emg_old, phase2_call_pose2emg_new)

phase2_call_emg2pose_old = """                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )"""
phase2_call_emg2pose_new = """                if item.get("our_pred_joints") is not None:
                    item["our_pred_joints"] = _pad_time_first(item["our_pred_joints"], max_t)
                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item.get("our_pred_joints"), item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )"""
content = content.replace(phase2_call_emg2pose_old, phase2_call_emg2pose_new)

# Oh wait, we need to fix the _compute_3d_bounds import logic or usage, but since I rewrote it locally inside render_emg2pose_new, we don't need it.

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

