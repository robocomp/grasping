import os
import argparse
import random
import json
import numpy as np
from skimage.io import imsave
from PIL import Image
from scipy.io import savemat
from math import tan, atan, radians, degrees

from utils import *
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

def simulate(scene_dir, cls_indices):
    # read 3d point cloud vertices npy (for visualization)
    vertex_npy = np.load("mesh_data/custom_vertex.npy")

    # loop over all scene files in scenes directory
    for scene_path in os.listdir(scene_dir):
        # check whether it's a scene file or not
        if not scene_path[-3:] == 'ttt':
            continue

        # create an output directory for each scene
        scene_out_dir = os.path.join('./sim_data/', scene_path[:-4])
        if not os.path.isdir(scene_out_dir):
            os.mkdir(scene_out_dir)

        # launch scene file
        SCENE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join(scene_dir, scene_path))
        pr = PyRep()
        pr.launch(SCENE_FILE, headless=True)
        pr.start()

        pr.step()

        # define vision sensor
        camera = VisionSensor('cam')
        # set camera absolute pose to world coordinates
        camera.set_pose([0,0,0,0,0,0,1])
        pr.step()

        # define background plane
        plane = Shape('Plane')
        plane.set_color([0, 0, 0])
        pr.step()

        # define scene shapes
        shapes = []
        shapes.append(Shape('Shape1'))
        shapes.append(Shape('Shape2'))
        shapes.append(Shape('Shape3'))
        shapes.append(Shape('Shape4'))
        pr.step()

        print("Getting vision sensor intrinsics ...")
        # get vision sensor parameters
        cam_res = camera.get_resolution()
        cam_per_angle = camera.get_perspective_angle()
        ratio = cam_res[0]/cam_res[1]
        cam_angle_x = 0.0
        cam_angle_y = 0.0
        if (ratio > 1):
            cam_angle_x = cam_per_angle
            cam_angle_y = 2 * degrees(atan(tan(radians(cam_per_angle / 2)) / ratio))
        else:
            cam_angle_x = 2 * degrees(atan(tan(radians(cam_per_angle / 2)) / ratio))
            cam_angle_y = cam_per_angle
        # get vision sensor intrinsic matrix
        cam_focal_x = float(cam_res[0] / 2.0) / tan(radians(cam_angle_x / 2.0))
        cam_focal_y = float(cam_res[1] / 2.0) / tan(radians(cam_angle_y / 2.0))
        intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res[0]/2.0)],
                                [0.00000000e+00, cam_focal_y, float(cam_res[1]/2.0)],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # loop to get 5000 samples per scene
        for i in range(1):
            print("Randomizing objects' poses ...")
            # set random pose to objects in the scene
            obj_colors = []
            for shape in shapes:
                shape.set_pose([
                        random.uniform(-2,2), random.uniform(-2,2), random.uniform(4,10),
                        random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1),
                        random.uniform(-1,1)
                    ])
                obj_colors.append([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
                pr.step()

            print("Reading vision sensor RGB signal ...")
            # read vision sensor RGB image
            img = camera.capture_rgb()
            img = np.uint8(img * 255.0)

            print("Reading vision sensor depth signal ...")
            # read vision sensor depth image
            depth = camera.capture_depth()
            depth = np.uint8(depth * 255.0)

            print("Generating front mask for RGB signal ...")
            # generate RGB front mask
            front_mask = np.sum(img, axis=2)
            front_mask[front_mask != 0] = 255
            front_mask = Image.fromarray(np.uint8(front_mask))
            alpha_img = Image.fromarray(img)
            alpha_img.putalpha(front_mask)

            print("Saving sensor output ...")
            # save sensor output
            alpha_img.save(scene_out_dir+f'/{str(i).zfill(6)}-color.png')
            imsave(scene_out_dir+f'/{str(i).zfill(6)}-depth.png', depth)
            
            print("Getting objects' poses ...")
            # get poses for all objects in scene
            poses = []
            for shape in shapes:
                poses.append(get_object_pose(shape, camera))
            pose_mat = np.stack(poses, axis=2)
            # save pose visualization on RGB image
            img_viz = visualize_predictions(poses, cls_indices[scene_path], img, vertex_npy, intrinsics)
            imsave(scene_out_dir+f'/{str(i).zfill(6)}-viz.png', img_viz)

            print("Saving meta-data ...")
            # save meta-data .mat
            meta_dict = {
                'cls_indexes'      : np.array(cls_indices[scene_path]),
                'intrinsic_matrix' : intrinsics,
                'poses'            : pose_mat
            }
            savemat(scene_out_dir+f'/{str(i).zfill(6)}-meta.mat', meta_dict)

            print("Performing semantic segmentation of RGB signal ...")
            # perform semantic segmentation of RGB image
            seg_img = camera.capture_rgb()
            seg_img = perform_segmentation(seg_img, cls_indices[scene_path], poses, vertex_npy, intrinsics)
            imsave(scene_out_dir+f'/{str(i).zfill(6)}-label.png', seg_img)

        # stop simulation
        pr.stop()
        pr.shutdown()

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-sd', '--scene_dir', type=str, help='path to the scenes directory', default='scenes/')
    argparser.add_argument('-cl', '--classes_json', type=str, help='json file containing ordered classes indices for each scene',
                        default='scenes/classes.json')

    args = argparser.parse_args()

    if not os.path.isdir('./sim_data/'):
        os.mkdir('./sim_data/')

    with open(args.classes_json) as file:
        cls_indices = json.load(file)

    simulate(args.scene_dir, cls_indices)
