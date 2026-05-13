import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# 1. Add our model loader and infer functions
our_funcs = """def _load_our_model(checkpoint_path: Path, device: Any) -> tuple[Any, Any]:
    from custom.tools.Mia_style_eval import _build_stage2_from_ckpt
    model, cfg, _, stage1 = _build_stage2_from_ckpt(checkpoint_path, device, stage1_override_ckpt=None)
    return model, stage1

def _infer_our_pose2emg(model, stage1, joints3d_t_25_3, condval, device) -> np.ndarray:
    import torch
    root = joints3d_t_25_3[:, 0:1, :]
    inputs_np = joints3d_t_25_3 - root
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(inputs, cond=cond)
        pred = out["pred"]
        stage1_emg = getattr(stage1, "emg", None)
        emg_standardizer = getattr(stage1_emg, "standardizer", None) if stage1_emg else None
        if emg_standardizer is not None:
            from custom.tools.Mia_style_eval import _emg_standardizer_stats_bt8
            mean_bt8, std_bt8 = _emg_standardizer_stats_bt8(emg_standardizer, t=int(pred.shape[1]), device=device)
            pred = pred * std_bt8 + mean_bt8
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32).T

def _infer_our_emg2pose(model, emg_8_t, condval, device, joints3d_t_25_3) -> np.ndarray:
    import torch
    inputs_np = emg_8_t.T
    inputs = torch.from_numpy(inputs_np).unsqueeze(0).to(device)
    cond = torch.tensor([[condval]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(inputs, cond=cond)
        pred = out["pred"]
    pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    root = joints3d_t_25_3[:, 0:1, :]
    return pred_np + root

"""
content = content.replace('def _load_model', our_funcs + 'def _load_model')

# 2. Modify cfg loading
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

# 3. Modify Phase 1 item preparation to store raw arrays
prep_old = """                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh,
                })
            else:
                if cache_file.exists():"""

prep_new = """                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh,
                    "raw_joints": arrays["joints3d_t_25_3"], "raw_emg": arrays["emg_8_t"]
                })
            else:
                if cache_file.exists():"""
content = content.replace(prep_old, prep_new)

prep2_old = """                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32)
                })"""

prep2_new = """                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32),
                    "raw_joints": arrays["joints3d_t_25_3"], "raw_emg": arrays["emg_8_t"]
                })"""
content = content.replace(prep2_old, prep2_new)

# 4. Insert Our Model Inference between Phase 1 and Global Filtering
our_infer = """
    # Run Our Model Inference on the selected items
    if our_checkpoint_path.exists():
        our_model, our_stage1 = _load_our_model(our_checkpoint_path, device)
        for exercise, (prepared, max_t) in all_prepared.items():
            for item in prepared:
                sample = item["sample"]
                temp_dir = out_dir / "temp" / sample.exercise
                our_cache_file = temp_dir / f"{sample.sample_id}_our_pred.npy"
                our_metric_file = temp_dir / f"{sample.sample_id}_our_metric.json"
                condval = _subject_to_condval(sample.subject)
                
                if task == "pose2emg":
                    if our_cache_file.exists():
                        our_pred_emg_8_t = np.load(our_cache_file)
                    else:
                        our_pred_emg_8_t = _infer_our_pose2emg(our_model, our_stage1, item["raw_joints"], condval, device)
                        np.save(our_cache_file, our_pred_emg_8_t)
                    
                    if not our_metric_file.exists():
                        gt_flat = item["raw_emg"].T
                        metric = _compute_official_metrics(our_pred_emg_8_t.T, gt_flat, task)
                        with open(our_metric_file, 'w') as mf: json.dump(metric, mf)
                        
                    our_pred_emg_mesh = _normalize_emg_for_mesh(our_pred_emg_8_t, sample.subject)
                    item["our_pred_emg_plot"] = our_pred_emg_8_t
                    item["our_pred_emg_mesh"] = our_pred_emg_mesh
                else:
                    if our_cache_file.exists():
                        our_pred_joints = np.load(our_cache_file)
                    else:
                        our_pred_joints = _infer_our_emg2pose(our_model, item["raw_emg"], condval, device, item["raw_joints"])
                        np.save(our_cache_file, our_pred_joints)
                        
                    if not our_metric_file.exists():
                        metric = _compute_official_metrics(our_pred_joints, item["raw_joints"][:, :25, :], task)
                        with open(our_metric_file, 'w') as mf: json.dump(metric, mf)
                        
                    item["our_pred_joints"] = our_pred_joints

"""
content = content.replace('    # Global Filtering', our_infer + '    # Global Filtering')

# 5. Modify rendering functions signature and logic
sig_p2e_old = """def _render_sequence_cells_pose2emg(
    renderer: Any, background_bgr: np.ndarray, verts_t_v_3: np.ndarray, origcam_t_4: np.ndarray,
    gt_emg_mesh_8_t: np.ndarray, pred_emg_mesh_8_t: np.ndarray, gt_emg_plot_8_t: np.ndarray, pred_emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, plot_vmax: float, mesh_views: str, debug_overlay_text: bool
) -> list[np.ndarray]:"""

sig_p2e_new = """def _render_sequence_cells_pose2emg(
    renderer: Any, background_bgr: np.ndarray, verts_t_v_3: np.ndarray, origcam_t_4: np.ndarray,
    gt_emg_mesh_8_t: np.ndarray, pred_emg_mesh_8_t: np.ndarray, our_emg_mesh_8_t: np.ndarray,
    gt_emg_plot_8_t: np.ndarray, pred_emg_plot_8_t: np.ndarray, our_emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, plot_vmax: float, mesh_views: str, debug_overlay_text: bool
) -> list[np.ndarray]:"""
content = content.replace(sig_p2e_old, sig_p2e_new)

body_p2e_old = """    panel_gt = cv2.cvtColor(_render_emg_panel(gt_emg_plot_8_t, plot_width, plot_height, "GT EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(_render_emg_panel(pred_emg_plot_8_t, plot_width, plot_height, "Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    for i in range(t):
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
        frames.append(np.concatenate([np.concatenate([gt_mesh, panel_gt], axis=1), np.concatenate([pred_mesh, panel_pred], axis=1)], axis=0))"""

body_p2e_new = """    panel_gt = cv2.cvtColor(_render_emg_panel(gt_emg_plot_8_t, plot_width, plot_height, "GT EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_pred = cv2.cvtColor(_render_emg_panel(pred_emg_plot_8_t, plot_width, plot_height, "Official Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    panel_our = cv2.cvtColor(_render_emg_panel(our_emg_plot_8_t, plot_width, plot_height, "Our Pred EMG", vmax=plot_vmax), cv2.COLOR_RGB2BGR)
    frames = []
    for i in range(t):
        verts, cam = verts_t_v_3[i], origcam_t_4[i]
        meshes = []
        for is_pred, emg_mesh in [(False, gt_emg_mesh_8_t), (True, pred_emg_mesh_8_t), (True, our_emg_mesh_8_t)]:
            views = []
            if mesh_views in ["front", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=True, pred=is_pred)
                views.append(v)
            if mesh_views in ["back", "both"]:
                v, _ = renderer.render(flag="False", current_path="/tmp/mia", img=background_bgr, verts=verts, emg_values=emg_mesh[:, i], cam=cam, front=False, pred=is_pred)
                views.append(v)
            meshes.append(np.concatenate(views, axis=1))
        
        gt_mesh, pred_mesh, our_mesh = meshes
        frames.append(np.concatenate([
            np.concatenate([gt_mesh, panel_gt], axis=1), 
            np.concatenate([pred_mesh, panel_pred], axis=1),
            np.concatenate([our_mesh, panel_our], axis=1)
        ], axis=0))"""
content = content.replace(body_p2e_old, body_p2e_new)

sig_e2p_old = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:"""

sig_e2p_new = """def _render_sequence_cells_emg2pose(
    gt_joints_t_25_3: np.ndarray, pred_joints_t_25_3: np.ndarray, our_joints_t_25_3: np.ndarray, emg_plot_8_t: np.ndarray,
    fps: int, plot_width: int, plot_height: int, render_width: int, render_height: int, plot_vmax: float, debug_overlay_text: bool
) -> list[np.ndarray]:"""
content = content.replace(sig_e2p_old, sig_e2p_new)

body_e2p_old = """    bounds = _compute_3d_bounds(gt_joints_t_25_3, pred_joints_t_25_3)
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Pred 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], pred_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Pred:Red)", bounds), cv2.COLOR_RGB2BGR)
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        overlay_row = np.concatenate([overlay_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row, overlay_row], axis=0))"""

body_e2p_new = """    bounds = _compute_3d_bounds(np.concatenate([gt_joints_t_25_3, pred_joints_t_25_3, our_joints_t_25_3], axis=0), np.zeros_like(gt_joints_t_25_3)) # HACK: compute_3d_bounds handles concatenated
    for i in tqdm(range(t), desc="Rendering emg2pose frames", leave=False):
        gt_skel = cv2.cvtColor(_render_skeleton_panel(gt_joints_t_25_3[i, :25], render_width, render_height, "GT 3D Pose", bounds), cv2.COLOR_RGB2BGR)
        pred_skel = cv2.cvtColor(_render_skeleton_panel(pred_joints_t_25_3[i, :25], render_width, render_height, "Official Pred", bounds), cv2.COLOR_RGB2BGR)
        our_skel = cv2.cvtColor(_render_skeleton_panel(our_joints_t_25_3[i, :25], render_width, render_height, "Our Pred", bounds), cv2.COLOR_RGB2BGR)
        overlay_skel = cv2.cvtColor(_render_overlay_skeleton_panel(gt_joints_t_25_3[i, :25], our_joints_t_25_3[i, :25], render_width, render_height, "Overlay (GT:Green, Our:Red)", bounds), cv2.COLOR_RGB2BGR)
        
        gt_row = np.concatenate([gt_skel, panel_emg], axis=1)
        pred_row = np.concatenate([pred_skel, panel_emg], axis=1)
        our_row = np.concatenate([our_skel, panel_emg], axis=1)
        overlay_row = np.concatenate([overlay_skel, panel_emg], axis=1)
        frames.append(np.concatenate([gt_row, pred_row, our_row, overlay_row], axis=0))"""
content = content.replace(body_e2p_old, body_e2p_new)

# 6. Update Phase 2 call
call_p2e_old = """                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )"""

call_p2e_new = """                item["our_pred_emg_plot"] = _pad_emg_8_t(item["our_pred_emg_plot"], max_t)
                item["our_pred_emg_mesh"] = _pad_emg_8_t(item["our_pred_emg_mesh"], max_t)
                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"], item["our_pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], item["our_pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )"""
content = content.replace(call_p2e_old, call_p2e_new)

call_e2p_old = """                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )"""

call_e2p_new = """                item["our_pred_joints"] = _pad_time_first(item["our_pred_joints"], max_t)
                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["our_pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )"""
content = content.replace(call_e2p_old, call_e2p_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

