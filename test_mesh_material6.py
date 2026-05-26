import pyrender
import trimesh
import numpy as np

mesh = trimesh.creation.icosphere()
mesh.visual.vertex_colors = np.array([[255, 0, 0, 255]] * len(mesh.vertices))

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))

m = pyrender.Mesh.from_trimesh(mesh)
for prim in m.primitives:
    prim.material.metallicFactor = 0.0
    prim.material.roughnessFactor = 1.0

scene.add(m)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light, pose=np.eye(4))

camera = pyrender.OrthographicCamera(xmag=1.5, ymag=1.5)
camera_pose = np.eye(4)
camera_pose[2, 3] = 3.0
scene.add(camera, pose=camera_pose)

r = pyrender.OffscreenRenderer(400, 400)
color, _ = r.render(scene)

import cv2
cv2.imwrite("test_mesh_material6.png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
