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

import vrepConst


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

        self.cameras = {}
        cam = VisionSensor("Camera_Shoulder")
        self.cameras["Camera_Shoulder"] = {"handle": cam, 
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
        self.grasping_objects["002_master_chef_can"] = {"handle": can,
                                                        "sim_pose": None,
                                                        "pred_pose_rgb": None,
                                                        "pred_pose_rgbd": None}

        with (open("objects_pcl.pickle", "rb")) as file:
            self.object_pcl = pickle.load(file)

        self.intrinsics = np.array([[self.cameras["Camera_Shoulder"]["focal"], 0.00000000e+00, self.cameras["Camera_Shoulder"]["width"]/2.0],
                                [0.00000000e+00, self.cameras["Camera_Shoulder"]["focal"], self.cameras["Camera_Shoulder"]["height"]/2.0],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.arm_ops = {"MoveToHome" : 1, 
                        "MoveToObj" : 2, 
                        "CloseGripper" : 3, 
                        "OpenGripper" : 4}

        self.arm_base = Shape("gen3")
        self.arm_target = Dummy("target")

    def compute(self):
        print('SpecificWorker.compute...')
        try:
            self.pr.step()

            # read arm camera RGB signal
            cam = self.cameras["Camera_Shoulder"]
            image_float = cam["handle"].capture_rgb()
            depth = cam["handle"].capture_depth(in_meters=False)
            image = cv2.normalize(src=image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cam["image_rgb"] = RoboCompObjectPoseEstimationRGB.TImage(width=cam["width"], height=cam["height"], depth=3, 
                                                                focalx=cam["focal"], focaly=cam["focal"], image=image.tobytes())
            cam["image_rgbd"] = RoboCompObjectPoseEstimationRGBD.TImage(width=cam["width"], height=cam["height"], depth=3, 
                                                                focalx=cam["focal"], focaly=cam["focal"], image=image.tobytes())
            cam["depth"] = RoboCompObjectPoseEstimationRGBD.TDepth(width=cam["width"], height=cam["height"], depthFactor=1.0, depth=depth.tobytes())

            # get objects's poses from simulator
            for obj_name in self.grasping_objects.keys():
                self.grasping_objects[obj_name]["sim_pose"] = self.grasping_objects[obj_name]["handle"].get_pose()
            
            # get objects' poses from RGB
            pred_poses = self.objectposeestimationrgb_proxy.getObjectPose(cam["image_rgb"])
            self.visualize_poses(image_float, pred_poses, "rgb_pose.png")
            for pose in pred_poses:
                if pose.objectname in self.grasping_objects.keys():
                    obj_trans = [pose.x, pose.y, pose.z + 0.2]
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

            # create a dummy for arm path planning
            approach_dummy = Dummy.create()
            approach_dummy.set_name("approach_dummy")
            approach_dummy.set_pose(self.grasping_objects["002_master_chef_can"]["pred_pose_rgbd"]) # NOTE : choose simulator or predicted pose

            # initialize approach dummy in embedded lua scripts
            call_ret = self.pr.script_call("initDummy@gen3", vrepConst.sim_scripttype_childscript)

            # move gen3 arm to the object
            self.move_arm(approach_dummy, self.arm_ops["MoveToObj"])

            # remove the created approach dummy
            approach_dummy.remove()
            
        except Exception as e:
            print(e)
        return True

    def process_pose(self, obj_trans, obj_rot):
        # convert an object pose from camera frame to world frame
        # define camera pose and z-axis flip matrix
        cam_trans = self.cameras["Camera_Shoulder"]["position"]
        cam_rot_mat = R.from_quat(self.cameras["Camera_Shoulder"]["rotation"])
        z_flip = R.from_matrix(np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
        # get object position in world coordinates
        obj_trans = np.dot(cam_rot_mat.as_matrix(), np.dot(z_flip.as_matrix(), np.array(obj_trans).reshape(-1,)))
        final_trans = obj_trans + cam_trans
        # get object orientation in world coordinates
        obj_rot_mat = R.from_quat(obj_rot)
        final_rot_mat = obj_rot_mat * z_flip * cam_rot_mat
        final_rot = final_rot_mat.as_quat()
        # return final object pose in world coordinates
        final_pose = list(final_trans)
        final_pose.extend(list(final_rot))
        return final_pose

    def visualize_poses(self, image, poses, img_name):
        # visualize the predicted poses on RGB image
        image = np.uint8(image*255.0)
        for pose in poses:
            # visualize only defined objects
            if pose.objectname not in self.grasping_objects.keys():
                continue
            obj_pcl = self.object_pcl[pose.objectname]
            obj_trans = np.array([pose.x, pose.y, pose.z])
            obj_rot = R.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
            proj_pcl = self.vertices_reprojection(obj_pcl, obj_rot, obj_trans, self.intrinsics)
            image = self.draw_pcl(image, proj_pcl, r=1, color=(randint(0,255), randint(0,255), randint(0,255)))
        cv2.imwrite(img_name, image)

    def vertices_reprojection(self, vertices, r, t, k):
        # re-project vertices in pixel space
        p = np.matmul(k, np.matmul(r, vertices.T) + t.reshape(-1,1))
        p[0] = p[0] / (p[2] + 1e-5)
        p[1] = p[1] / (p[2] + 1e-5)
        return p[:2].T

    def draw_pcl(self, img, p2ds, r=1, color=(255, 0, 0)):
        # draw object point cloud on RGB image
        h, w = img.shape[0], img.shape[1]
        for pt_2d in p2ds:
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            img = cv2.circle(img, (int(pt_2d[0]), int(pt_2d[1])), r, color, -1)
        return img

    def move_arm(self, dummy_dest, func_number):
        # move arm to destination
        # NOTE : this function is using remote lua scripts embedded in the arm
        # for better path planning, so make sure to use the correct arm model
        call_function = True
        init_pose = np.array(self.arm_target.get_pose(relative_to=self.arm_base))
        # loop until the arm reached the object
        while True:
            # step the simulation
            self.pr.step()
            # set function index to the desired operation
            if call_function:
                try:
                    # call thearded child lua scripts via PyRep
                    call_ret = self.pr.script_call("setFunction@gen3", vrepConst.sim_scripttype_childscript, ints=[func_number])
                except Exception as e:
                    print(e)
            # get current poses to compare
            actual_pose = self.arm_target.get_pose(relative_to=self.arm_base)
            object_pose = dummy_dest.get_pose(relative_to=self.arm_base)
            # compare poses to check for operation end
            pose_diff = np.abs(np.array(actual_pose) - np.array(init_pose))
            if(call_function and pose_diff[0] > 0.01 or pose_diff[1] > 0.01 or pose_diff[2] > 0.01):
                call_function = False
            # check whether the arm reached the target
            dest_pose_diff = np.abs(np.array(actual_pose) - np.array(object_pose))
            if(dest_pose_diff[0] < 0.015 and dest_pose_diff[1] < 0.015 and dest_pose_diff[2] < 0.015):
                break

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
