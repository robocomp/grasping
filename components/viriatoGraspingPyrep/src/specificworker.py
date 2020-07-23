#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2020 by Mohamed Shawky
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from genericworker import *
import os, time, queue
from bisect import bisect_left
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.gen3 import Gen3
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.end_effectors.mico_gripper import MicoGripper
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape

import numpy as np
from random import randint
import cv2
import queue
import pickle
from scipy.spatial.transform import Rotation as R


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        super(SpecificWorker, self).__init__(proxy_map)

    def __del__(self):
        print('SpecificWorker destructor')
        self.pr.stop()
        self.pr.shutdown()

    def setParams(self, params):
        SCENE_FILE = params["scene_dir"]
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()

        self.gen3_arm = Gen3()
        self.mico_gripper = MicoGripper(0)

        self.cameras = {}
        cam = VisionSensor("Gen3_depth_sensor")
        self.cameras["Gen3_depth_sensor"] = {"handle": cam, 
                                            "id": 1,
                                            "angle": np.radians(cam.get_perspective_angle()), 
                                            "width": cam.get_resolution()[0],
                                            "height": cam.get_resolution()[1],
                                            "depth": 3,
                                            "focal": cam.get_resolution()[0]/np.tan(np.radians(cam.get_perspective_angle())), 
                                            "position": cam.get_position(), 
                                            "rotation": cam.get_quaternion(), 
                                            "image_rgb": np.array(0),
                                            "image_rgbd": np.array(0),
                                            "depth": np.ndarray(0)}

        self.grasping_objects = {}
        can = Shape("can")
        self.grasping_objects["002_master_chef_can"] = {"handler": can,
                                                        "sim_pose": None,
                                                        "pred_pose_rgb": None,
                                                        "pred_pose_rgbd": None}

        with (open("objects_pcl.pickle", "rb")) as file:
            self.object_pcl = pickle.load(file)

        self.intrinsics = np.array([[self.cameras["Gen3_depth_sensor"]["focal"], 0.00000000e+00, self.cameras["Gen3_depth_sensor"]["width"]/2.0],
                                [0.00000000e+00, self.cameras["Gen3_depth_sensor"]["focal"], self.cameras["Gen3_depth_sensor"]["height"]/2.0],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    def compute(self):
        print('SpecificWorker.compute...')
        while True:
            try:
                self.pr.step()

                # read arm camera RGB signal
                cam = self.cameras["Gen3_depth_sensor"]
                image_float = cam["handle"].capture_rgb()
                depth = cam["handle"].capture_depth()
                image = cv2.normalize(src=image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cam["image_rgb"] = RoboCompObjectPoseEstimationRGB.TImage(width=cam["width"], height=cam["height"], depth=3, 
                                                                    focalx=cam["focal"], focaly=cam["focal"], image=image.tobytes())
                cam["image_rgbd"] = RoboCompObjectPoseEstimationRGBD.TImage(width=cam["width"], height=cam["height"], depth=3, 
                                                                    focalx=cam["focal"], focaly=cam["focal"], image=image.tobytes())
                cam["depth"] = RoboCompObjectPoseEstimationRGBD.TDepth(width=cam["width"], height=cam["height"], depth=depth.tobytes())

                # get objects's poses from simulator
                for obj_name in self.grasping_objects.keys():
                    self.grasping_objects[obj_name]["sim_pose"] = self.grasping_objects[obj_name]["handler"].get_pose()
                
                # get objects' poses from RGB
                pred_poses = self.objectposeestimationrgb_proxy.getObjectPose(cam["image_rgb"])
                self.visualize_poses(image_float, pred_poses, "rgb_pose.png")
                for pose in pred_poses:
                    if pose.objectname in self.grasping_objects.keys():
                        obj_trans = [pose.x, pose.y, pose.z]
                        obj_quat = [pose.qx, pose.qy, pose.qz, pose.qw]
                        obj_pose = self.process_pose(obj_trans, obj_quat)
                        self.grasping_objects[pose.objectname]["pred_pose_rgb"] = obj_pose

                # get objects' poses from RGBD
                pred_poses = self.objectposeestimationrgbd_proxy.getObjectPose(cam["image_rgbd"], cam["depth"])
                self.visualize_poses(image_float, pred_poses, "rgbd_pose.png")
                for pose in pred_poses:
                    if pose.objectname in self.grasping_objects.keys():
                        obj_trans = [pose.x, pose.y, pose.z]
                        obj_quat = [pose.qx, pose.qy, pose.qz, pose.qw]
                        obj_pose = self.process_pose(obj_trans, obj_quat)
                        self.grasping_objects[pose.objectname]["pred_pose_rgbd"] = obj_pose
                
            except Exception as e:
                print(e)
        return True

    def process_pose(self, obj_trans, obj_rot):
        # convert an object pose from camera frame to world frame
        final_trans = obj_trans + self.cameras["Gen3_depth_sensor"]["position"]
        cam_rot_mat = R.from_quat(self.cameras["Gen3_depth_sensor"]["rotation"]).as_matrix()
        obj_rot_mat = R.from_quat(obj_rot).as_matrix()
        final_rot_mat = np.matmul(obj_rot_mat, cam_rot_mat)
        final_rot = R.from_matrix(final_rot_mat).as_quat()
        final_pose = list(final_trans)
        final_pose.extend(list(final_rot))
        return final_pose

    def visualize_poses(self, image, poses, img_name):
        image = np.uint8(image*255.0)
        for pose in poses:
            if pose.objectname not in self.grasping_objects.keys():
                continue
            obj_pcl = self.object_pcl[pose.objectname]
            obj_trans = np.array([pose.x, pose.y, pose.z])
            obj_rot = R.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
            proj_pcl = self.vertices_reprojection(obj_pcl, obj_rot, obj_trans, self.intrinsics)
            image = self.draw_pcl(image, proj_pcl, r=1, color=(randint(0,255), randint(0,255), randint(0,255)))
        cv2.imwrite(img_name, image)

    def vertices_reprojection(self, vertices, r, t, k):
        p = np.matmul(k, np.matmul(r, vertices.T) + t.reshape(-1,1))
        p[0] = p[0] / (p[2] + 1e-5)
        p[1] = p[1] / (p[2] + 1e-5)
        return p[:2].T

    def draw_pcl(self, img, p2ds, r=1, color=(255, 0, 0)):
        h, w = img.shape[0], img.shape[1]
        for pt_2d in p2ds:
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            img = cv2.circle(img, (int(pt_2d[0]), int(pt_2d[1])), r, color, -1)
        return img

    ######################
    # From the RoboCompObjectPoseEstimationRGB you can call this methods:
    # self.objectposeestimationrgb_proxy.getObjectPose(...)

    ######################
    # From the RoboCompObjectPoseEstimationRGB you can use this types:
    # RoboCompObjectPoseEstimationRGB.TImage
    # RoboCompObjectPoseEstimationRGB.ObjectPose

    ######################
    # From the RoboCompObjectPoseEstimationRGBD you can call this methods:
    # self.objectposeestimationrgbd_proxy.getObjectPose(...)

    ######################
    # From the RoboCompObjectPoseEstimationRGBD you can use this types:
    # RoboCompObjectPoseEstimationRGBD.TImage
    # RoboCompObjectPoseEstimationRGBD.TDepth
    # RoboCompObjectPoseEstimationRGBD.ObjectPose
