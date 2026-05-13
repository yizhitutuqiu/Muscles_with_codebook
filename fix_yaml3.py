with open('/data/litengmo/HSMR/mia_custom/custom/vis/configs/vis_infer_final.yaml', 'r') as f:
    content = f.read()

content = content.replace("n_per_exercise: 1", "n_per_exercise: 3")
content = content.replace("max_exercises: 1", "max_exercises: -1")
content = content.replace("filter_worst_n: 1", "filter_worst_n: 3")

with open('/data/litengmo/HSMR/mia_custom/custom/vis/configs/vis_infer_final.yaml', 'w') as f:
    f.write(content)
