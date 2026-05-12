import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Change sampling logic to load ALL samples if filter_worst_n > 0
sampling_old = """    selected_by_exercise = {}
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
        random.shuffle(cand)
        selected_by_exercise[ex] = cand[:n_per_exercise]"""

sampling_new = """    selected_by_exercise = {}
    filter_worst_n = cfg.get("filter_worst_n", 0)
    for ex in exercise_names:
        cand = samples_by_exercise[ex]
        if filter_worst_n > 0:
            # Load ALL samples for inference to find the global worst
            selected_by_exercise[ex] = cand
        else:
            random.shuffle(cand)
            selected_by_exercise[ex] = cand[:n_per_exercise]"""

content = content.replace(sampling_old, sampling_new)

# Remove the old per-exercise filtering logic
phase2_old = """    # Phase 2: Visualization
    for exercise, (prepared, max_t) in tqdm(sorted(all_prepared.items(), key=lambda x: x[0]), desc="Phase 2: Visualization"):
        if filter_worst_n > 0 and len(prepared) > filter_worst_n:
            # Read metrics and sort descending
            scored_prepared = []
            temp_dir = out_dir / "temp" / exercise
            for item in prepared:
                metric_file = temp_dir / f"{item['sample'].sample_id}_metric.json"
                rmse = 0.0
                if metric_file.exists():
                    with open(metric_file, 'r') as mf:
                        m = json.load(mf)
                        rmse = m.get("official_global_rmse", 0.0)
                scored_prepared.append((rmse, item))
            scored_prepared.sort(key=lambda x: x[0], reverse=True)
            prepared = [x[1] for x in scored_prepared[:filter_worst_n]]
            
        t1 = time.time()
        cell_streams = []"""

phase2_new = """    # Global Filtering
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
            all_prepared = {}

    # Phase 2: Visualization
    for exercise, (prepared, max_t) in tqdm(sorted(all_prepared.items(), key=lambda x: x[0]), desc="Phase 2: Visualization"):
        t1 = time.time()
        cell_streams = []"""

content = content.replace(phase2_old, phase2_new)

# Update the writer path logic for global
writer_old = """        if filter_worst_n > 0:
            vid_out_dir = out_dir / "selected" / exercise
        else:
            vid_out_dir = out_dir / exercise"""

writer_new = """        if filter_worst_n > 0:
            vid_out_dir = out_dir / "selected"
        else:
            vid_out_dir = out_dir / exercise"""

content = content.replace(writer_old, writer_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

