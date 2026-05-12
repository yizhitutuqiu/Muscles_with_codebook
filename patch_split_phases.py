import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

old_loop = """    total_inference_time = 0.0
    total_visualization_time = 0.0
    for exercise, samples_in_ex in tqdm(sorted(selected_by_exercise.items(), key=lambda x: x[0]), desc="Processing exercises"):
        prepared = []
        max_t = 0
        for sample in samples_in_ex:
            arrays = _load_sample_arrays(sample)
            condval = _subject_to_condval(sample.subject)
            t = arrays["emg_8_t"].shape[1]
            max_t = max(max_t, t)
            
            # Setup temp cache directory
            temp_dir = out_dir / "temp" / exercise
            temp_dir.mkdir(parents=True, exist_ok=True)
            cache_file = temp_dir / f"{sample.sample_id}_pred.npy"
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
                })

        if not prepared: continue

        t1 = time.time()
        cell_streams = []
        for item in prepared:
            if task == "pose2emg":
                item["verts"] = _pad_time_first(item["verts"], max_t)
                item["cam"] = _pad_time_first(item["cam"], max_t)
                item["gt_emg_plot"] = _pad_emg_8_t(item["gt_emg_plot"], max_t)
                item["pred_emg_plot"] = _pad_emg_8_t(item["pred_emg_plot"], max_t)
                item["gt_emg_mesh"] = _pad_emg_8_t(item["gt_emg_mesh"], max_t)
                item["pred_emg_mesh"] = _pad_emg_8_t(item["pred_emg_mesh"], max_t)
                
                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )
            else:
                item["gt_joints"] = _pad_time_first(item["gt_joints"], max_t)
                item["pred_joints"] = _pad_time_first(item["pred_joints"], max_t)
                item["emg_plot"] = _pad_emg_8_t(item["emg_plot"], max_t)
                
                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )
            cell_streams.append(cell_frames)

        cell_h, cell_w = cell_streams[0][0].shape[:2]
        writer = _open_video_writer(out_dir / exercise / f"grid_{task}_n{len(prepared)}.mp4", fps, (cell_w * len(cell_streams), cell_h))
        try:
            for i in range(max_t):
                writer.write(np.concatenate([cell_streams[c][i] for c in range(len(cell_streams))], axis=1))
        finally:
            writer.release()
        total_visualization_time += time.time() - t1"""

new_loop = """    total_inference_time = 0.0
    total_visualization_time = 0.0
    
    # Phase 1: Inference & Caching
    all_prepared = {}
    for exercise, samples_in_ex in tqdm(sorted(selected_by_exercise.items(), key=lambda x: x[0]), desc="Phase 1: Inference & Cache"):
        prepared = []
        max_t = 0
        for sample in samples_in_ex:
            arrays = _load_sample_arrays(sample)
            condval = _subject_to_condval(sample.subject)
            t = arrays["emg_8_t"].shape[1]
            max_t = max(max_t, t)
            
            # Setup temp cache directory
            temp_dir = out_dir / "temp" / exercise
            temp_dir.mkdir(parents=True, exist_ok=True)
            cache_file = temp_dir / f"{sample.sample_id}_pred.npy"
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
                })

        if prepared:
            all_prepared[exercise] = (prepared, max_t)

    # Phase 2: Visualization
    for exercise, (prepared, max_t) in tqdm(sorted(all_prepared.items(), key=lambda x: x[0]), desc="Phase 2: Visualization"):
        t1 = time.time()
        cell_streams = []
        for item in prepared:
            if task == "pose2emg":
                item["verts"] = _pad_time_first(item["verts"], max_t)
                item["cam"] = _pad_time_first(item["cam"], max_t)
                item["gt_emg_plot"] = _pad_emg_8_t(item["gt_emg_plot"], max_t)
                item["pred_emg_plot"] = _pad_emg_8_t(item["pred_emg_plot"], max_t)
                item["gt_emg_mesh"] = _pad_emg_8_t(item["gt_emg_mesh"], max_t)
                item["pred_emg_mesh"] = _pad_emg_8_t(item["pred_emg_mesh"], max_t)
                
                cell_frames = _render_sequence_cells_pose2emg(
                    renderer, background, item["verts"], item["cam"], item["gt_emg_mesh"], item["pred_emg_mesh"],
                    item["gt_emg_plot"], item["pred_emg_plot"], fps, plot_width, plot_height, plot_emg_vmax, mesh_views, debug_overlay_text
                )
            else:
                item["gt_joints"] = _pad_time_first(item["gt_joints"], max_t)
                item["pred_joints"] = _pad_time_first(item["pred_joints"], max_t)
                item["emg_plot"] = _pad_emg_8_t(item["emg_plot"], max_t)
                
                cell_frames = _render_sequence_cells_emg2pose(
                    item["gt_joints"], item["pred_joints"], item["emg_plot"], fps, plot_width, plot_height,
                    render_width, render_height, plot_emg_vmax, debug_overlay_text
                )
            cell_streams.append(cell_frames)

        cell_h, cell_w = cell_streams[0][0].shape[:2]
        writer = _open_video_writer(out_dir / exercise / f"grid_{task}_n{len(prepared)}.mp4", fps, (cell_w * len(cell_streams), cell_h))
        try:
            for i in range(max_t):
                writer.write(np.concatenate([cell_streams[c][i] for c in range(len(cell_streams))], axis=1))
        finally:
            writer.release()
        total_visualization_time += time.time() - t1"""

content = content.replace(old_loop, new_loop)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
