import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# 1. Read new config parameter
cfg_old = """    filter_our_max_rmse_threshold = cfg.get("filter_our_max_rmse_threshold", 999.0)"""
cfg_new = """    filter_our_max_rmse_threshold = cfg.get("filter_our_max_rmse_threshold", 999.0)
    filter_diversity_exercise = cfg.get("filter_diversity_exercise", False)"""
content = content.replace(cfg_old, cfg_new, 1)

# 2. Modify Global Filtering logic
filter_old = """        # Sort globally descending
        all_items_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        target_n = filter_our_best_diff_n if filter_our_best_diff_n > 0 else filter_worst_n
        selected_items = [x[1] for x in all_items_with_scores[:target_n]]"""

filter_new = """        # Sort globally descending
        all_items_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        target_n = filter_our_best_diff_n if filter_our_best_diff_n > 0 else filter_worst_n
        selected_items = []
        if filter_diversity_exercise:
            seen_exercises = set()
            for score, item in all_items_with_scores:
                ex = item["sample"].exercise
                if ex not in seen_exercises:
                    selected_items.append(item)
                    seen_exercises.add(ex)
                if len(selected_items) >= target_n:
                    break
        else:
            selected_items = [x[1] for x in all_items_with_scores[:target_n]]"""

content = content.replace(filter_old, filter_new)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

