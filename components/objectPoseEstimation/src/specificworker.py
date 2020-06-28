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


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# sys.path.append('/opt/robocomp/lib')
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

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
                                '061_foam_brick', 'custom-fork-01', 'custom-fork-02', 'custom-glass-01',
                                'custom-jar-01', 'custom-jar-02', 'custom-plate-01', 'custom-plate-02',
                                'custom-plate-03', 'custom-spoon-01', 'custom-spoon-02']
            # define point cloud vertices of used models
            self.vertices = np.load(params["vertices_file"])
            # configure network
            self.model = configure_network(cfg_file=params["config_file"], weights_file=params["weights_file"])
            # initialize predicted poses
            self.final_poses = []
        except:
            print("Error reading config params")
        return True


    @QtCore.Slot()
    def compute(self):
        try:
            # get RGB image information
            img_buffer = self.camerargbdsimple_proxy.getImage(self.camera_name)
            image = np.frombuffer(img_buffer.image, np.uint8).reshape(img_buffer.height, img_buffer.width, img_buffer.depth)
            # get vision sensor intrinstic parameters
            cam_res_x = img_buffer.width
            cam_res_y = img_buffer.height
            cam_focal_x = img_buffer.focalx
            cam_focal_y = img_buffer.focaly
            intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res_x/2.0)],
                                    [0.00000000e+00, cam_focal_y, float(cam_res_y/2.0)],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            # perform network inference
            pred_poses = get_pose(self.model, image, self.class_names, intrinsics, self.vertices, save_results=False)
            # post-process network output
            self.final_poses = self.process_poses(pred_poses)
            # publish predicted poses
            self.objectposeestimationpub_proxy.pushObjectPose(RoboCompObjectPoseEstimation.PoseType(self.final_poses))
        except Ice.Exception as e:
            print(e)
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
            # get euler angles for rotation
            rot_mat = pose[1][:3,0:3]
            rot = R.from_matrix(rot_mat)
            rot_euler = rot.as_euler('xyz')
            # build object pose type
            obj_pose = RoboCompObjectPoseEstimation.ObjectPose(objectname=object_name,
                                                                x=trans_mat[0],
                                                                y=trans_mat[1],
                                                                z=trans_mat[2],
                                                                rx=rot_euler[0],
                                                                ry=rot_euler[1],
                                                                rz=rot_euler[2])
            prc_poses.append(obj_pose)
        return prc_poses


    # =============== Methods for Component Implements ==================
    # ===================================================================

    #
    # IMPLEMENTATION of getObjectPose method from ObjectPoseEstimation interface
    #
    def ObjectPoseEstimation_getObjectPose(self, img):
        # extract RGB image
        image = np.frombuffer(img.image, np.uint8).reshape(img.height, img.width, img.depth)
        # get vision sensor intrinstic parameters
        cam_res_x = img.width
        cam_res_y = img.height
        cam_focal_x = img.focalx
        cam_focal_y = img.focaly
        intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res_x/2.0)],
                                [0.00000000e+00, cam_focal_y, float(cam_res_y/2.0)],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # perform network inference
        pred_poses = get_pose(self.model, image, self.class_names, intrinsics, self.vertices, save_results=False)
        # post-process network output
        ret_poses = self.process_poses(pred_poses)
        # publish predicted poses
        return RoboCompObjectPoseEstimation.PoseType(self.final_poses)

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
    # From the RoboCompObjectPoseEstimation you can use this types:
    # RoboCompObjectPoseEstimation.TImage
    # RoboCompObjectPoseEstimation.ObjectPose

