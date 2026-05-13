import json
import sys
from pathlib import Path

out_dir = Path("/data/litengmo/HSMR/mia_custom/custom/output/vis_infer_final_emg2pose")

scores = []
for temp_dir in out_dir.glob("temp/*"):
    for metric_file in temp_dir.glob("*_metric.json"):
        if "our" in metric_file.name: continue
        
        sample_id = metric_file.name.replace("_metric.json", "")
        our_metric_file = temp_dir / f"{sample_id}_our_metric.json"
        
        with open(metric_file, 'r') as mf:
            official_rmse = json.load(mf).get("official_global_rmse", 0.0)
            
        our_rmse = 0.0
        if our_metric_file.exists():
            with open(our_metric_file, 'r') as mf:
                our_rmse = json.load(mf).get("official_global_rmse", 0.0)
                
        score = official_rmse - our_rmse
        scores.append((score, official_rmse, our_rmse, temp_dir.name, sample_id))

scores.sort(key=lambda x: x[0], reverse=True)

filter_diversity_exercise = True
seen_exercises = set()
filtered_scores = []

for s in scores:
    exercise = s[3]
    if filter_diversity_exercise:
        if exercise in seen_exercises:
            continue
        seen_exercises.add(exercise)
    filtered_scores.append(s)

for i in range(min(5, len(filtered_scores))):
    print(f"Rank {i+1}: Exercise {filtered_scores[i][3]} Sample {filtered_scores[i][4]} | Score(Diff): {filtered_scores[i][0]:.4f} | Official: {filtered_scores[i][1]:.4f} | Our: {filtered_scores[i][2]:.4f}")


