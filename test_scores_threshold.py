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
                
        if our_rmse > 0.1: continue
                
        score = official_rmse - our_rmse
        scores.append((score, official_rmse, our_rmse, temp_dir.name, sample_id))

scores.sort(key=lambda x: x[0], reverse=True)
for i in range(min(5, len(scores))):
    print(f"Rank {i+1}: Exercise {scores[i][3]} Sample {scores[i][4]} | Score(Diff): {scores[i][0]:.4f} | Official: {scores[i][1]:.4f} | Our: {scores[i][2]:.4f}")

