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
import numpy_indexed as npi
from itertools import zip_longest
import cv2
import queue


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map):
        super(SpecificWorker, self).__init__(proxy_map)

    def __del__(self):
        print('SpecificWorker destructor')

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
                                            "rgb": np.array(0), 
                                            "depth": np.ndarray(0)}

        self.grasping_objects = {}
        self.grasping_objects["002_master_chef_can"] = Shape("can")


    def compute(self):
        print('SpecificWorker.compute...')
        while True:
            try:
                self.pr.step()
                cam = self.cameras["Gen3_depth_sensor"]
                image_float = cam["handle"].capture_rgb()
                depth = cam["handle"].capture_depth()
                image = cv2.normalize(src=image_float, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cam["rgb"] = RoboCompObjectPoseEstimation.TImage(width=cam["width"], height=cam["height"], depth=3, focalx=cam["focal"], focaly=cam["focal"], image=image.tobytes())



    ######################
    # From the RoboCompObjectPoseEstimation you can call this methods:
    # self.objectposeestimation_proxy.getObjectPose(...)

    ######################
    # From the RoboCompObjectPoseEstimation you can use this types:
    # RoboCompObjectPoseEstimation.TImage
    # RoboCompObjectPoseEstimation.ObjectPose

