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
from pose_estimator import *
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
            # define classes names
            self.class_names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
                                '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
                                '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp',
                                '061_foam_brick', 'custom-can-01', 'custom-fork-01', 'custom-fork-02',
                                'custom-glass-01', 'custom-jar-01', 'custom-knife-01', 'custom-plate-01',
                                'custom-plate-02', 'custom-plate-03', 'custom-spoon-01']
            # define point cloud vertices of used models
            self.vertices = np.load(params["vertices_file"])
            # configure network
            self.model = configure_network(cfg_file=params["config_file"], weights_file=params["weights_file"])
            # define calibartion offset along camera z-axis
            self.z_offset = float(params["cam_z_offset"])
            # set save visualizations boolean
            self.save_viz = True if params["save_viz"].lower() == 'true' else False
            # initialize predicted poses
            self.final_poses = []
        except Exception as e:
            print("Error reading config params")
            print(e)
            return False
        return True

    @QtCore.Slot()
    def compute(self):
        print('SpecificWorker.compute...')
        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    def process_poses(self, pred_poses):
        prc_poses = []
        # loop over each predicted pose
        for pose in pred_poses:
            # get class name
            object_name = self.class_names[pose[0]]
            # get translation matrix
            trans_mat = pose[1][:3,3]
            # add calibration offset along camera z-axis
            trans_mat[2] += self.z_offset
            # get quaternions for rotation
            rot_mat = pose[1][:3,0:3]
            rot = R.from_matrix(rot_mat)
            rot_quat = rot.as_quat()
            # build object pose type
            obj_pose = RoboCompObjectPoseEstimationRGB.ObjectPose(objectname=object_name,
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
    # IMPLEMENTATION of getObjectPose method from ObjectPoseEstimationRGB interface
    #
    def ObjectPoseEstimationRGB_getObjectPose(self, image):
        # extract RGB image
        img = np.frombuffer(image.image, np.uint8).reshape(image.height, image.width, image.depth)
        # get vision sensor intrinstic parameters
        cam_res_x = image.width
        cam_res_y = image.height
        cam_focal_x = image.focalx
        cam_focal_y = image.focaly
        intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res_x/2.0)],
                                [0.00000000e+00, cam_focal_y, float(cam_res_y/2.0)],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # perform network inference
        pred_poses = get_pose(self.model, img, self.class_names, intrinsics, self.vertices, save_results=self.save_viz)
        # post-process network output
        ret_poses = self.process_poses(pred_poses)
        # publish predicted poses
        return RoboCompObjectPoseEstimationRGB.PoseType(ret_poses)

    # ===================================================================
    # ===================================================================

    ######################
    # From the RoboCompObjectPoseEstimationRGB you can use this types:
    # RoboCompObjectPoseEstimationRGB.ObjectPose
