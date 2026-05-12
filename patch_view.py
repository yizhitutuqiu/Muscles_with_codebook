with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'r') as f:
    content = f.read()

content = content.replace('ax.view_init(0, 180)', 'ax.view_init(0, -90)')

with open('/data/litengmo/HSMR/mia_custom/custom/vis/vis_infer_final.py', 'w') as f:
    f.write(content)
