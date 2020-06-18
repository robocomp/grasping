import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from math import tan, atan, radians, degrees, sqrt

####################################################################################################################
#                                               Visualization Utilities                                            #
####################################################################################################################
def get_color_by_index(idx):
    # return a specific color for each new class
    classes_colors = {
        22 : [255, 0, 0],   # fork1 -> red
        23 : [0, 0, 255],   # fork2 -> blue
        24 : [255, 255, 0], # glass1 -> yellow
        25 : [255, 165, 0], # jar1 -> orange
        26 : [165, 42, 42], # jar2 -> brown
        27 : [128, 0, 128], # plate1 -> purple
        28 : [0, 255, 255], # plate2 -> cyan
        29 : [0, 255, 0],   # plate3 -> green
        30 : [128, 128, 0], # spoon1 -> olive
        31 : [255, 0, 255]  # spoon2 -> magenta 
    }
    return classes_colors[idx]

def vertices_reprojection(vertices, rt, k):
    # project a vertex to pixel space
    p = np.matmul(k, np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def visualize_predictions(pred_pose, cls_idx, image, vertex, intrinsics):
    # visualize estimated poses on RGB image
    height, width, _ = image.shape
    mask_img = np.zeros((height,width), np.uint8)
    contour_img = np.copy(image)
    for i in range(len(pred_pose)):
        # show surface reprojection
        mask_img.fill(0)
        vp = vertices_reprojection(vertex[cls_idx[i]-1][:], pred_pose[i], intrinsics)
        for p in vp:
            if p[0] != p[0] or p[1] != p[1]:  # check nan
                continue
            mask_img = cv2.circle(mask_img, (int(p[0]), int(p[1])), 1, 255, -1)

        # fill the holes
        kernel = np.ones((5,5), np.uint8)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(mask_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 4, cv2.LINE_AA) # border
        contour_img = cv2.drawContours(contour_img, contours, -1, get_color_by_index(cls_idx[i]), 2, cv2.LINE_AA)

    return contour_img


####################################################################################################################
#                                               GT Poses Utilities                                                 #
####################################################################################################################
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


####################################################################################################################
#                                               Segmentation Utilities                                             #
####################################################################################################################
def project_poses(pred_pose, cls_idx, image, vertex, intrinsics):
    # project estimated poses on RGB image (to be used in segmentation)
    height, width, depth = image.shape
    mask_img = np.zeros((height,width), np.uint8)
    contour_img = np.zeros((height,width,depth), np.uint8)
    for i in range(len(pred_pose)):
        # show surface reprojection
        mask_img.fill(0)
        vp = vertices_reprojection(vertex[cls_idx[i]-1][:], pred_pose[i], intrinsics)
        for p in vp:
            if p[0] != p[0] or p[1] != p[1]:  # check nan
                continue
            mask_img = cv2.circle(mask_img, (int(p[0]), int(p[1])), 1, 255, -1)

        # fill the holes
        kernel = np.ones((5,5), np.uint8)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(mask_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(contour_img, contours, -1, get_color_by_index(cls_idx[i]), 2, cv2.LINE_AA)
        # fill contour
        contour_img = cv2.fillPoly(contour_img, pts =contours, color=get_color_by_index(cls_idx[i]))

    return contour_img

def get_color_index(obj_colors, pixel_color):
    # get index of the equivalent to pixel color
    for i in range(len(obj_colors)):
        if (obj_colors[i]==pixel_color).all():
            return i
    return -1

def perform_segmentation(img, cls_idx, poses, vertices, intrinsics):
    # initialize segmentation mask
    height, width, _ = img.shape
    obj_colors = [get_color_by_index(cls_idx[0]),
                get_color_by_index(cls_idx[1]),
                get_color_by_index(cls_idx[2]),
                get_color_by_index(cls_idx[3])]
    init_mask = project_poses(poses, cls_idx, np.uint8(img * 255.0), vertices, intrinsics)
    seg_mask = np.zeros((height, width), np.uint8)
    # loop over each pixel
    for i in range(height):
        for j in range(width):
            if (init_mask[i][j]==np.array([0,0,0])).all():
                continue
            near_idx = get_color_index(obj_colors, init_mask[i][j])
            if near_idx == -1:
                continue
            seg_mask[i][j] = cls_idx[near_idx]
    return seg_mask
