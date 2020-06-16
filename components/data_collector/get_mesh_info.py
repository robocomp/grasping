import os
import argparse
import numpy as np
import trimesh

def augment_gt_data(bbox_npy, vertex_npy):
    """
    Produces mesh data information npys 
    Returns centred bounding boxes and mesh point cloud vertices
    Arguments:
    bbox_npy   : path to original bounding boxes npy.
    vertex_npy : path to original vertices npy. 
    """
    # load original npys
    full_bbox_arr = np.load(bbox_npy)
    full_vertex_arr = np.load(vertex_npy)

    # attach trimesh to console logger
    trimesh.util.attach_to_log()

    # set maximum number of vertices per mesh
    max_vertex = full_vertex_arr.shape[1]
    vertices = []

    # loop over all meshes
    for filename in os.listdir('meshes'):
        print(f'Getting information of {filename} ...')

        print(f'Loading mesh ...')
        mesh = trimesh.load('meshes/'+filename)

        print(f'Saving bounding box vertices ...')
        obj_bbox = mesh.bounding_box.vertices.reshape((1, 8, 3))
        full_bbox_arr = np.concatenate((full_bbox_arr, obj_bbox), axis=0)
        np.save('mesh_data/custom_bbox.npy', full_bbox_arr)

        print(f'Storing 3D point cloud of shape {mesh.vertices.shape} (to be written) ...')
        max_vertex = mesh.vertices.shape[0] if (mesh.vertices.shape[0] > max_vertex) else max_vertex
        vertices.append(mesh.vertices)

    # pad 3d point cloud vertices for all meshes by zeros (for all to have the same shape)
    print(f'Saving new set of 3D point cloud vertices ...')
    original_vertices_shape = full_vertex_arr.shape
    padded_vertices_arr = np.zeros((original_vertices_shape[0], max_vertex, original_vertices_shape[2]))
    padded_vertices_arr[:,:original_vertices_shape[1],:] = full_vertex_arr
    for obj_vertex in vertices:
        padded_obj_vertex = np.zeros((1, max_vertex, 3))
        padded_obj_vertex[:,:obj_vertex.shape[0],:] = obj_vertex.reshape((1, obj_vertex.shape[0], obj_vertex.shape[1]))
        padded_vertices_arr = np.concatenate((padded_vertices_arr, padded_obj_vertex), axis=0)
    np.save('mesh_data/custom_vertex.npy', padded_vertices_arr)

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-bbp', '--bbox_path', type=str, help='path to original dataset bounding boxes', 
                            default='../../segmentation-based-pose/configs/YCB-Video/YCB_bbox.npy')
    argparser.add_argument('-vp', '--vertices_path', type=str, help='path to original dataset point cloud vertices', 
                            default='../../segmentation-based-pose/configs/YCB-Video/YCB_vertex.npy')

    args = argparser.parse_args()

    if not os.path.isdir("./mesh_data/"):
        os.mkdir("./mesh_data/")

    augment_gt_data(args.bbox_path, args.vertices_path)
