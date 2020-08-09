import os
import argparse
import json
import numpy as np
import trimesh
import open3d as o3d

def augment_gt_data(bbox_npy, vertex_npy, cls_indices):
    """
    Produces mesh data information npys 
    Returns centred bounding boxes and mesh point cloud vertices
    Arguments:
    bbox_npy    : path to original bounding boxes npy.
    vertex_npy  : path to original vertices npy.
    cls_indices : dictionary of new classes indices.
    """
    # load original npys
    full_bbox_arr = np.load(bbox_npy)
    full_vertex_arr = np.load(vertex_npy)

    # attach trimesh to console logger
    trimesh.util.attach_to_log()

    # loop over all meshes
    for filename in cls_indices.keys():
        print(f'Getting information of {filename} ...')

        print(f'Loading mesh ...')
        tm_mesh = trimesh.load('meshes/custom/'+filename+'/'+filename+'.obj')
        o3d_mesh = o3d.io.read_triangle_mesh('meshes/custom/'+filename+'/'+filename+'.obj')

        print(f'Saving bounding box vertices of size (8,3) ...')
        obj_bbox = tm_mesh.bounding_box.vertices.reshape((1, 8, 3))
        full_bbox_arr = np.concatenate((full_bbox_arr, obj_bbox), axis=0)

        print(f'Saving 3D point cloud vertices of size (10000,3) ...')
        obj_pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
        obj_pcd = np.asarray(obj_pcd.points).reshape((1, 10000, 3))
        full_vertex_arr = np.concatenate((full_vertex_arr, obj_pcd), axis=0)

    # save output bounding boxes
    print(f'Writing new bounding box vertices ...')
    np.save('mesh_data/custom_bbox.npy', full_bbox_arr)

    # save point cloud vertices
    print(f'Writing new 3D point cloud vertices ...')
    np.save('mesh_data/custom_vertex.npy', full_vertex_arr)

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-bbp', '--bbox_path', type=str, help='path to original dataset bounding boxes', 
                        default='../segmentation-based-pose/configs/YCB-Video/YCB_bbox.npy')
    argparser.add_argument('-vp', '--vertices_path', type=str, help='path to original dataset point cloud vertices', 
                        default='../segmentation-based-pose/configs/YCB-Video/YCB_vertex.npy')
    argparser.add_argument('-cl', '--classes_json', type=str, help='json file indices for new classes',
                        default='meshes/custom/new_classes.json')

    args = argparser.parse_args()

    if not os.path.isdir("./mesh_data/"):
        os.mkdir("./mesh_data/")

    with open(args.classes_json) as file:
        cls_indices = json.load(file)

    augment_gt_data(args.bbox_path, args.vertices_path, cls_indices)
