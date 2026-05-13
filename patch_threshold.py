import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# 1. Read threshold from config
cfg_old = """    filter_our_best_diff_n = cfg.get("filter_our_best_diff_n", 0)"""
cfg_new = """    filter_our_best_diff_n = cfg.get("filter_our_best_diff_n", 0)
    filter_our_max_rmse_threshold = cfg.get("filter_our_max_rmse_threshold", 999.0)"""
content = content.replace(cfg_old, cfg_new, 1)

# 2. Add to global filtering logic
filter_old = """                if our_metric_file.exists():
                    with open(our_metric_file, 'r') as mf:
                        m = json.load(mf)
                        our_rmse = m.get("official_global_rmse", 0.0)
                
                if filter_our_best_diff_n > 0:
                    score = official_rmse - our_rmse # Higher is better (we beat official by more)
                else:
                    score = official_rmse # Higher is worse
                    
                all_items_with_scores.append((score, item))"""

filter_new = """                if our_metric_file.exists():
                    with open(our_metric_file, 'r') as mf:
                        m = json.load(mf)
                        our_rmse = m.get("official_global_rmse", 0.0)
                
                # Apply the max threshold filter
                if filter_our_max_rmse_threshold < 999.0 and our_rmse > filter_our_max_rmse_threshold:
                    continue
                
                if filter_our_best_diff_n > 0:
                    score = official_rmse - our_rmse # Higher is better (we beat official by more)
                else:
                    score = official_rmse # Higher is worse
                    
                all_items_with_scores.append((score, item))"""

content = content.replace(filter_old, filter_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

