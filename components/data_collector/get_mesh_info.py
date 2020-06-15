import os
import numpy as np
import trimesh

trimesh.util.attach_to_log()

for filename in os.listdir('meshes'):
    print(filename)
    mesh = trimesh.load('meshes/'+filename)
    mesh.bounding_box.extents
    print(mesh.bounding_box.vertices)
    print(mesh.vertices.shape)
