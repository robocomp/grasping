import os
import argparse
import random
import json
import cv2
import numpy as np
from skimage.io import imsave
from scipy.io import savemat
from scipy.spatial.transform import Rotation
from math import tan, atan, radians, degrees

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

def vertices_reprojection(vertices, rt, k):
    # project a vertex to pixel space
    p = np.matmul(k, np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def visualize_predictions(predPose, cls_idx, image, vertex, intrinsics):
    # visualize estimated poses on RGB image
    height, width, _ = image.shape
    maskImg = np.zeros((height,width), np.uint8)
    contourImg = np.copy(image)
    for i in range(len(predPose)):
        # show surface reprojection
        maskImg.fill(0)
        vp = vertices_reprojection(vertex[cls_idx[i]-1][:], predPose[i], intrinsics)
        for p in vp:
            if p[0] != p[0] or p[1] != p[1]:  # check nan
                continue
            maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 1, 255, -1)

        # fill the holes
        kernel = np.ones((5,5), np.uint8)
        maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contourImg = cv2.drawContours(contourImg, contours, -1, (255, 255, 255), 4, cv2.LINE_AA) # border
        contourImg = cv2.drawContours(contourImg, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)

    return contourImg

def get_object_pose(obj, cam):
    # get pose matrix of a shape using its position and quaternion
    # get translation (x & y axes are flipped)
    obj_position = obj.get_position(relative_to=cam).reshape((-1,1))
    obj_position[0] = obj_position[0] * -1
    obj_position[1] = obj_position[1] * -1

    # get rotation from quaternion (rotated 180 around z axis)
    obj_quat = obj.get_quaternion(relative_to=cam)
    obj_rot = Rotation.from_quat(obj_quat)
    flip_mat = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    obj_rot_mat = np.matmul(flip_mat, obj_rot.as_matrix())

    # concatenate rotation and translation matrices 
    obj_pose_mat = np.concatenate((obj_rot_mat, obj_position), 1)
    
    return obj_pose_mat

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

        # loop to get 100 samples per scene
        for i in range(100):
            print("Randomizing objects' poses and colors ...")
            # set random pose and color to objects in the scene
            for shape in shapes:
                shape.set_pose([
                        random.uniform(-2,2), random.uniform(-2,2), random.uniform(4,10),
                        random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1),
                        random.uniform(-1,1)
                    ])
                shape.set_color([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
                pr.step()

            print("Reading vision sensor RGB signal ...")
            # read vision sensor RGB image
            img = camera.capture_rgb()
            img = np.uint8(img * 255.0)

            print("Reading vision sensor depth signal ...")
            # read vision sensor depth image
            depth = camera.capture_depth()
            depth = np.uint8(depth * 255.0)

            print("Saving sensor output ...")
            # save sensor output
            imsave(scene_out_dir+f'/img_{i}.png', img)
            imsave(scene_out_dir+f'/depth_{i}.png', depth)
            
            print("Getting objects' poses ...")
            # get poses for all objects in scene
            poses = []
            for shape in shapes:
                poses.append(get_object_pose(shape, camera))
            pose_mat = np.stack(poses, axis=2)
            # save pose visualization on RGB image
            img_viz = visualize_predictions(poses, cls_indices[scene_path], img, vertex_npy, intrinsics)
            imsave(scene_out_dir+f'/viz_{i}.png', img_viz)

            print("Saving meta-data ...")
            # save meta-data .mat
            meta_dict = {
                'cls_indexes'      : np.array(cls_indices[scene_path]),
                'intrinsic_matrix' : intrinsics,
                'poses'            : pose_mat
            }
            savemat(scene_out_dir+f'/meta_{i}.mat', meta_dict)

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
