import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
import numpy as np
import cv2
import json
import matplotlib
import matplotlib as mpl
import sys
sys.path.append(".")
from smplx import SMPL

def colorFader(mix=0):
    c1=np.array(matplotlib.colors.to_rgb('white'))
    c2=np.array(matplotlib.colors.to_rgb('red'))
    return (1-mix)*c1 + mix*c2

def part_segm_to_vertex_colors(part_segm, n_vertices, front, emg_values, emg_vmax=0.5):
    vertex_colors = np.ones((n_vertices, 4)) * 150.0
    vertex_colors[:, 3] = 255.0
    
    norm = mpl.colors.Normalize(vmin=0, vmax=emg_vmax)
    emg_values = np.array(norm(emg_values.tolist()).tolist())
    emg_values[emg_values > 1.0] = 1.0
    emg_values[emg_values < 0.0] = 0.0
    
    def get_color(x):
        c = colorFader(x) * 255.0
        return [c[0], c[1], c[2], 255.0]
        
    for k, v in part_segm.items():
        if front:
            if k == 'rightUpLeg': vertex_colors[v] = get_color(emg_values[4])
            elif k == 'leftUpLeg': vertex_colors[v] = get_color(emg_values[0])
            elif k == 'leftArm': vertex_colors[v] = get_color(emg_values[3])
            elif k == 'rightArm': vertex_colors[v] = get_color(emg_values[7])
        else:
            if k == 'rightUpLeg': vertex_colors[v] = get_color(emg_values[5])
            elif k == 'leftUpLeg': vertex_colors[v] = get_color(emg_values[1])
            elif k == 'leftShoulder': vertex_colors[v] = get_color(emg_values[2])
            elif k == 'rightShoulder': vertex_colors[v] = get_color(emg_values[6])
                
    return vertex_colors

smpl = SMPL('musclesinaction/vibe_data', batch_size=1, create_transl=False)
faces = smpl.faces
part_segm = json.load(open('musclesinaction/smpl_vert_segmentation.json'))
sample_dir = "/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val/Subject0/Squat/114"
verts = np.load(os.path.join(sample_dir, "verts.npy"))
emg = np.load(os.path.join(sample_dir, "emgvalues.npy"))
if emg.shape[0] == 8: emg = emg.T

v = verts[0].copy()
vc = part_segm_to_vertex_colors(part_segm, v.shape[0], True, emg[0], emg_vmax=0.01)
vc = np.clip(vc, 0, 255).astype(np.uint8)

mesh = trimesh.Trimesh(v, faces, process=False, vertex_colors=vc)
Rx = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
mesh.apply_transform(Rx)

scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
m = pyrender.Mesh.from_trimesh(mesh)
scene.add(m)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light, pose=np.eye(4))

camera = pyrender.OrthographicCamera(xmag=mesh.extents[0]/2*1.1, ymag=mesh.extents[1]/2*1.1)
camera_pose = np.eye(4)
camera_pose[:3, 3] = mesh.centroid
camera_pose[2, 3] += mesh.extents[2] + 2.0
scene.add(camera, pose=camera_pose)

r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)
cv2.imwrite("test_mesh_render_debug.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
print("Saved test_mesh_render_debug.png")
