import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Fix config reading
cfg_old = """    filter_worst_n = cfg.get("filter_worst_n", 0)"""
cfg_new = """    filter_worst_n = cfg.get("filter_worst_n", 0)
    filter_our_best_diff_n = cfg.get("filter_our_best_diff_n", 0)"""
content = content.replace(cfg_old, cfg_new, 1)

# Fix early filtering logic, it should load ALL samples if either filter is > 0
early_filter_old = """    filter_worst_n = cfg.get("filter_worst_n", 0)
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
        if filter_worst_n > 0:
            # Load ALL samples for inference to find the global worst
            selected_by_exercise[ex] = cand"""
early_filter_new = """    filter_worst_n = cfg.get("filter_worst_n", 0)
    filter_our_best_diff_n = cfg.get("filter_our_best_diff_n", 0)
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
        if filter_worst_n > 0 or filter_our_best_diff_n > 0:
            # Load ALL samples for inference to find the global extreme
            selected_by_exercise[ex] = cand"""
content = content.replace(early_filter_old, early_filter_new)

# Modify model loading to load our_model before Phase 1
load_model_old = """    model = _load_model(checkpoint_path, device, task)

    total_inference_time = 0.0"""
load_model_new = """    model = _load_model(checkpoint_path, device, task)
    our_model, our_stage1 = None, None
    if our_checkpoint_path.exists():
        our_model, our_stage1 = _load_our_model(our_checkpoint_path, device)

    total_inference_time = 0.0"""
content = content.replace(load_model_old, load_model_new)


# In Phase 1 loop, we remove the separate "Run Our Model Inference" block and integrate it directly into Phase 1 loop.
# Wait, actually it's easier to just keep the separate block, but just run it ALWAYS during Phase 1 if our_model is loaded.
# The current code already runs it on `all_prepared`, which contains ALL samples if filtering is > 0.
# So "Run Our Model Inference on the selected items" is already running on ALL samples if filtering is on!
# I just need to update the Global Filtering logic to support filter_our_best_diff_n.

global_filter_old = """    # Global Filtering
    if filter_worst_n > 0:
        # Collect all prepared items across all exercises
        all_items_with_scores = []
        for exercise, (prepared, _) in all_prepared.items():
            temp_dir = out_dir / "temp" / exercise
            for item in prepared:
                metric_file = temp_dir / f"{item['sample'].sample_id}_metric.json"
                rmse = 0.0
                if metric_file.exists():
                    with open(metric_file, 'r') as mf:
                        m = json.load(mf)
                        rmse = m.get("official_global_rmse", 0.0)
                all_items_with_scores.append((rmse, item))
        
        # Sort globally descending
        all_items_with_scores.sort(key=lambda x: x[0], reverse=True)
        global_worst_items = [x[1] for x in all_items_with_scores[:filter_worst_n]]
        
        # Repackage into a single pseudo-exercise called "global_worst"
        if global_worst_items:
            global_max_t = max(item["emg_plot"].shape[1] if task != "pose2emg" else item["gt_emg_plot"].shape[1] for item in global_worst_items)
            all_prepared = {"global_worst": (global_worst_items, global_max_t)}
        else:
            all_prepared = {}"""

global_filter_new = """    # Global Filtering
    if filter_worst_n > 0 or filter_our_best_diff_n > 0:
        all_items_with_scores = []
        for exercise, (prepared, _) in all_prepared.items():
            temp_dir = out_dir / "temp" / exercise
            for item in prepared:
                # Get Official RMSE
                metric_file = temp_dir / f"{item['sample'].sample_id}_metric.json"
                official_rmse = 0.0
                if metric_file.exists():
                    with open(metric_file, 'r') as mf:
                        m = json.load(mf)
                        official_rmse = m.get("official_global_rmse", 0.0)
                
                # Get Our RMSE
                our_metric_file = temp_dir / f"{item['sample'].sample_id}_our_metric.json"
                our_rmse = 0.0
                if our_metric_file.exists():
                    with open(our_metric_file, 'r') as mf:
                        m = json.load(mf)
                        our_rmse = m.get("official_global_rmse", 0.0)
                
                if filter_our_best_diff_n > 0:
                    score = official_rmse - our_rmse # Higher is better (we beat official by more)
                else:
                    score = official_rmse # Higher is worse
                    
                all_items_with_scores.append((score, item))
        
        # Sort globally descending
        all_items_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        target_n = filter_our_best_diff_n if filter_our_best_diff_n > 0 else filter_worst_n
        selected_items = [x[1] for x in all_items_with_scores[:target_n]]
        
        # Repackage into a single pseudo-exercise
        if selected_items:
            pseudo_name = "global_best_diff" if filter_our_best_diff_n > 0 else "global_worst"
            global_max_t = max(item["emg_plot"].shape[1] if task != "pose2emg" else item["gt_emg_plot"].shape[1] for item in selected_items)
            all_prepared = {pseudo_name: (selected_items, global_max_t)}
        else:
            all_prepared = {}"""
content = content.replace(global_filter_old, global_filter_new)

# Fix video writer path output
writer_old = """        if filter_worst_n > 0:
            vid_out_dir = out_dir / "selected"
        else:
            vid_out_dir = out_dir / exercise"""
writer_new = """        if filter_worst_n > 0 or filter_our_best_diff_n > 0:
            vid_out_dir = out_dir / "selected"
        else:
            vid_out_dir = out_dir / exercise"""
content = content.replace(writer_old, writer_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

