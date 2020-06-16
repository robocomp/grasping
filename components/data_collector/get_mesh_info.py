import os
import numpy as np
import trimesh

if not os.path.isdir("./mesh_data/"):
        os.mkdir("./mesh_data/")

trimesh.util.attach_to_log()

for filename in os.listdir('meshes'):
    print(f'Getting information of {filename} ...')

    print(f'Loading mesh ...')
    mesh = trimesh.load('meshes/'+filename)

    print(f'Saving 3D point cloud of shape {mesh.vertices.shape} ...')
    np.save(f'mesh_data/{filename[:-4]}_vertex.npy', mesh.vertices)

    print(f'Saving bouding box vertices of shape {mesh.bounding_box.vertices.shape} ...')
    np.save(f'mesh_data/{filename[:-4]}_bbox.npy', mesh.bounding_box.vertices)
