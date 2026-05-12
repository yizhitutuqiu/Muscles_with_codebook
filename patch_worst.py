import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Add filter_worst_n parsing
cfg_load_old = """    max_exercises = cfg.get("max_exercises", -1)"""
cfg_load_new = """    max_exercises = cfg.get("max_exercises", -1)
    filter_worst_n = cfg.get("filter_worst_n", 0)"""
content = content.replace(cfg_load_old, cfg_load_new)

# Modify Phase 2 logic to sort and filter by RMSE
phase2_old = """    # Phase 2: Visualization
    for exercise, (prepared, max_t) in tqdm(sorted(all_prepared.items(), key=lambda x: x[0]), desc="Phase 2: Visualization"):
        t1 = time.time()
        cell_streams = []"""

phase2_new = """    # Phase 2: Visualization
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

content = content.replace(phase2_old, phase2_new)

# Modify output directory based on filter_worst_n
writer_old = """        writer = _open_video_writer(out_dir / exercise / f"grid_{task}_n{len(prepared)}.mp4", fps, (cell_w * len(cell_streams), cell_h))"""
writer_new = """        
        if filter_worst_n > 0:
            vid_out_dir = out_dir / "selected" / exercise
        else:
            vid_out_dir = out_dir / exercise
        
        writer = _open_video_writer(vid_out_dir / f"grid_{task}_n{len(prepared)}.mp4", fps, (cell_w * len(cell_streams), cell_h))"""

content = content.replace(writer_old, writer_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

