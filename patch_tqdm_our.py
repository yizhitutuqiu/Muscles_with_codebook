import re

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

# Add tqdm to our model inference
old_loop = """    # Run Our Model Inference on the selected items
    if our_checkpoint_path.exists():
        our_model, our_stage1 = _load_our_model(our_checkpoint_path, device)
        for exercise, (prepared, max_t) in all_prepared.items():
            for item in prepared:"""

new_loop = """    # Run Our Model Inference on the selected items (Phase 1.5)
    if our_checkpoint_path.exists():
        if our_model is None:
            our_model, our_stage1 = _load_our_model(our_checkpoint_path, device)
        for exercise, (prepared, max_t) in tqdm(all_prepared.items(), desc="Phase 1.5: Our Model Inference & Cache"):
            for item in prepared:"""

content = content.replace(old_loop, new_loop)

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)

