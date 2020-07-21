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

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from genericworker import *
from dnn.pvn3d.api import *
import numpy as np
from scipy.spatial.transform import Rotation as R


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = 2000
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        print('SpecificWorker destructor')

    def setParams(self, params):
        try:
            # define camera handler to stream from
            self.camera_name = params["camera_name"]
            # define classes names
            self.class_names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
                                '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
                                '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp',
                                '061_foam_brick', 'custom-can-01', 'custom-fork-01', 'custom-fork-02',
                                'custom-glass-01', 'custom-jar-01', 'custom-knife-01', 'custom-plate-01',
                                'custom-plate-02', 'custom-plate-03', 'custom-spoon-01']
            # configure network
            self.pose_estimator = RGBDPoseAPI(weights_path=params["weights_file"])
            # initialize predicted poses
            self.final_poses = []
        except:
            print("Error reading config params")
            return False
        return True


    @QtCore.Slot()
    def compute(self):
        print('SpecificWorker.compute...')
        try:
            # get RGBD image information
            rgbd_buffer = self.camerargbdsimple_proxy.getAll(self.camera_name)
            img_buffer = rgbd_buffer.image
            depth_buffer = rgbd_buffer.depth
            image = np.frombuffer(img_buffer.image, np.uint8).reshape(img_buffer.height, img_buffer.width, img_buffer.depth)
            depth = np.frombuffer(depth_buffer.depth, np.float32).reshape(depth_buffer.height, depth_buffer.width)
            depth = np.uint8(depth * 255.0)
            # get vision sensor intrinstic parameters
            cam_res_x = img_buffer.width
            cam_res_y = img_buffer.height
            cam_focal_x = img_buffer.focalx
            cam_focal_y = img_buffer.focaly
            intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res_x/2.0)],
                                    [0.00000000e+00, cam_focal_y, float(cam_res_y/2.0)],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            # pre-process RGBD data
            self.pose_estimator.preprocess_rgbd(image, depth, intrinsics)
            # perform network inference
            pred_cls, pred_poses = self.pose_estimator.get_poses(save_results=False)
            # post-process network output
            self.final_poses = self.process_poses(pred_cls, pred_poses)
            # publish predicted poses
            self.objectposeestimationpub_proxy.pushObjectPose(RoboCompObjectPoseEstimationRGBD.PoseType(self.final_poses))
        except Exception as e:
            print(e)
            return False
        return True

        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def process_poses(self, pred_cls, pred_poses):
        prc_poses = []
        # loop over each predicted pose
        for cls, pose in zip(pred_cls, pred_poses):
            # get class name
            object_name = self.class_names[cls-1]
            # get translation matrix
            trans_mat = pose[:3,3]
            # get quaternions for rotation
            rot_mat = pose[:3,0:3]
            rot = R.from_matrix(rot_mat)
            rot_quat = rot.as_quat()
            # build object pose type
            obj_pose = RoboCompObjectPoseEstimationRGBD.ObjectPose(objectname=object_name,
                                                                x=trans_mat[0],
                                                                y=trans_mat[1],
                                                                z=trans_mat[2],
                                                                qx=rot_quat[0],
                                                                qy=rot_quat[1],
                                                                qz=rot_quat[2],
                                                                qw=rot_quat[3])
            prc_poses.append(obj_pose)
        return prc_poses


    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of getObjectPose method from ObjectPoseEstimationRGBD interface
    #
    def ObjectPoseEstimationRGBD_getObjectPose(self, image, depth):
        # extract RGB image
        img = np.frombuffer(image.image, np.uint8).reshape(image.height, image.width, image.depth)
        dep = np.frombuffer(depth.depth, np.float32).reshape(depth.height, depth.width)
        dep = np.uint8(dep * 255.0)
        # get vision sensor intrinstic parameters
        cam_res_x = image.width
        cam_res_y = image.height
        cam_focal_x = image.focalx
        cam_focal_y = image.focaly
        intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res_x/2.0)],
                                [0.00000000e+00, cam_focal_y, float(cam_res_y/2.0)],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # pre-process RGBD data
        self.pose_estimator.preprocess_rgbd(img, dep, intrinsics)
        # perform network inference
        pred_cls, pred_poses = self.pose_estimator.get_poses(save_results=False)
        # post-process network output
        ret_poses = self.process_poses(pred_cls, pred_poses)
        # return predicted poses
        return RoboCompObjectPoseEstimationRGBD.PoseType(ret_poses)
    # ===================================================================
    # ===================================================================


    ######################
    # From the RoboCompCameraRGBDSimple you can call this methods:
    # self.camerargbdsimple_proxy.getAll(...)
    # self.camerargbdsimple_proxy.getDepth(...)
    # self.camerargbdsimple_proxy.getImage(...)

    ######################
    # From the RoboCompCameraRGBDSimple you can use this types:
    # RoboCompCameraRGBDSimple.TImage
    # RoboCompCameraRGBDSimple.TDepth
    # RoboCompCameraRGBDSimple.TRGBD

    ######################
    # From the RoboCompObjectPoseEstimationPub you can publish calling this methods:
    # self.objectposeestimationpub_proxy.pushObjectPose(...)

    ######################
    # From the RoboCompObjectPoseEstimationRGBD you can use this types:
    # RoboCompObjectPoseEstimationRGBD.TImage
    # RoboCompObjectPoseEstimationRGBD.TDepth
    # RoboCompObjectPoseEstimationRGBD.ObjectPose

