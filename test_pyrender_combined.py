import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyrender
import trimesh
import numpy as np
import cv2
import json

import matplotlib
import matplotlib as mpl

def colorFader(mix=0):
    c1=np.array(matplotlib.colors.to_rgb('white'))
    c2=np.array(matplotlib.colors.to_rgb('red'))
    return (1-mix)*c1 + mix*c2

def part_segm_to_vertex_colors(part_segm, n_vertices, front, emg_values):
    vertex_colors = np.ones((n_vertices, 4))
    norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
    emg_values = np.array(norm(emg_values.tolist()).tolist())
    emg_values[emg_values > 1.0]=1.0
    emg_values[emg_values < 0.0]=0.0
    emg_values = emg_values.tolist()
    colorfader = []
    for x in emg_values:
        out = colorFader(x)
        colorfader.append([out[0], out[1], out[2], 1.0])
        
    for part_idx, (k, v) in enumerate(part_segm.items()):
        if front:
            if 'front' in k:
                for vertex_idx in v:
                    vertex_colors[vertex_idx] = colorfader[part_idx]
        else:
            if 'back' in k:
                for vertex_idx in v:
                    vertex_colors[vertex_idx] = colorfader[part_idx]
    return vertex_colors * 255.0

import sys
sys.path.append(".")
from smplx import SMPL

smpl = SMPL('musclesinaction/vibe_data', batch_size=1, create_transl=False)
faces = smpl.faces

sample_dir = "/data/litengmo/HSMR/mia_custom/MIADatasetOfficial/val/Subject0/Squat/114"
verts = np.load(os.path.join(sample_dir, "verts.npy"))
emg = np.load(os.path.join(sample_dir, "emgvalues.npy"))

if emg.shape[0] == 8: emg = emg.T
part_segm = json.load(open('musclesinaction/smpl_vert_segmentation.json'))

mesh_list = []
n_frames = 5
T = verts.shape[0]
indices = np.linspace(0, T - 1, n_frames, dtype=int)

min_x, max_x = np.min(verts[:, :, 0]), np.max(verts[:, :, 0])
step_x = (max_x - min_x) * 0.7 if max_x > min_x else 0.3

for rank, idx in enumerate(indices):
    v = verts[idx].copy()
    v[:, 0] += step_x * rank
    
    vc = part_segm_to_vertex_colors(part_segm, v.shape[0], True, emg[idx])
    vc = np.clip(vc, 0, 255).astype(np.uint8)
    
    mesh = trimesh.Trimesh(v, faces, process=False, vertex_colors=vc)
    mesh_list.append(mesh)

combined = trimesh.util.concatenate(mesh_list)
Rx = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
combined.apply_transform(Rx)

scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
m = pyrender.Mesh.from_trimesh(combined)
scene.add(m)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light, pose=np.eye(4))

bounds = combined.bounds
centroid = combined.centroid
extents = combined.extents

camera = pyrender.OrthographicCamera(xmag=extents[0]/2 * 1.1, ymag=extents[1]/2 * 1.1)
camera_pose = np.eye(4)
camera_pose[:3, 3] = centroid
camera_pose[2, 3] += extents[2] + 2.0

scene.add(camera, pose=camera_pose)

r = pyrender.OffscreenRenderer(1200, 400)
color, depth = r.render(scene)
cv2.imwrite("test_mesh_render_rotated.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
print("Saved test_mesh_render_rotated.png")
