import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Add JSON import if not present
if 'import json' not in content:
    content = content.replace('import yaml', 'import json\nimport yaml')

# Add metric computation function
metric_func = """
def _compute_official_metrics(pred: np.ndarray, gt: np.ndarray, task: str) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    
    if task == "pose2emg":
        sq_err = np.square(pred - gt)
        overall_mse = float(np.mean(sq_err))
        overall_rmse = float(np.sqrt(overall_mse))
        return {"official_global_rmse": overall_rmse}
    else:  # emg2pose
        diff = pred - gt
        l2_dist = np.sqrt(np.sum(np.square(diff), axis=-1))
        mpjpe = float(np.mean(l2_dist))
        return {"official_global_rmse": mpjpe}
"""
if '_compute_official_metrics' not in content:
    content = content.replace('def _subject_to_condval', metric_func + '\ndef _subject_to_condval')

# Modify the inference section to support cache and metric calculation
cache_logic_old = """        for sample in samples_in_ex:
            arrays = _load_sample_arrays(sample)
            condval = _subject_to_condval(sample.subject)
            t = arrays["emg_8_t"].shape[1]
            max_t = max(max_t, t)
            
            if task == "pose2emg":
                t0 = time.time()
                pred_emg_8_t = _infer_emg(model, arrays["joints3d_t_25_3"], condval, device)
                total_inference_time += time.time() - t0
                gt_emg_mesh = _normalize_emg_for_mesh(arrays["emg_8_t"], sample.subject)
                pred_emg_mesh = _normalize_emg_for_mesh(pred_emg_8_t, sample.subject)
                prepared.append({
                    "sample": sample, "verts": arrays["verts_t_v_3"], "cam": arrays["origcam_t_4"],
                    "gt_emg_plot": arrays["emg_8_t"].astype(np.float32), "pred_emg_plot": pred_emg_8_t,
                    "gt_emg_mesh": gt_emg_mesh, "pred_emg_mesh": pred_emg_mesh,
                })
            else:
                t0 = time.time()
                pred_joints = _infer_pose(model, arrays["emg_8_t"], condval, device)
                total_inference_time += time.time() - t0
                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32)
                })"""

cache_logic_new = """        for sample in samples_in_ex:
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
                    metric = _compute_official_metrics(pred_joints, arrays["joints3d_t_25_3"], task)
                    with open(metric_file, 'w') as mf: json.dump(metric, mf)
                    
                prepared.append({
                    "sample": sample, "gt_joints": arrays["joints3d_t_25_3"], "pred_joints": pred_joints,
                    "emg_plot": arrays["emg_8_t"].astype(np.float32)
                })"""

content = content.replace(cache_logic_old, cache_logic_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

