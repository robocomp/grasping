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

import sys, Ice, os

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'

Ice.loadSlice("-I ./src/ --all ./src/CommonBehavior.ice")
import RoboCompCommonBehavior

Ice.loadSlice("-I ./src/ --all ./src/ObjectPoseEstimationRGB.ice")
import RoboCompObjectPoseEstimationRGB
Ice.loadSlice("-I ./src/ --all ./src/ObjectPoseEstimationRGBD.ice")
import RoboCompObjectPoseEstimationRGBD

class ImgType(list):
    def __init__(self, iterable=list()):
        super(ImgType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(ImgType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, byte)
        super(ImgType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(ImgType, self).insert(index, item)

setattr(RoboCompObjectPoseEstimationRGB, "ImgType", ImgType)

class PoseType(list):
    def __init__(self, iterable=list()):
        super(PoseType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompObjectPoseEstimationRGB.ObjectPose)
        super(PoseType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompObjectPoseEstimationRGB.ObjectPose)
        super(PoseType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompObjectPoseEstimationRGB.ObjectPose)
        super(PoseType, self).insert(index, item)

setattr(RoboCompObjectPoseEstimationRGB, "PoseType", PoseType)

class ImgType(list):
    def __init__(self, iterable=list()):
        super(ImgType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(ImgType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, byte)
        super(ImgType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(ImgType, self).insert(index, item)

setattr(RoboCompObjectPoseEstimationRGBD, "ImgType", ImgType)

class DepthType(list):
    def __init__(self, iterable=list()):
        super(DepthType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(DepthType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, byte)
        super(DepthType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(DepthType, self).insert(index, item)

setattr(RoboCompObjectPoseEstimationRGBD, "DepthType", DepthType)

class PoseType(list):
    def __init__(self, iterable=list()):
        super(PoseType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompObjectPoseEstimationRGBD.ObjectPose)
        super(PoseType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompObjectPoseEstimationRGBD.ObjectPose)
        super(PoseType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompObjectPoseEstimationRGBD.ObjectPose)
        super(PoseType, self).insert(index, item)

setattr(RoboCompObjectPoseEstimationRGBD, "PoseType", PoseType)


class GenericWorker():

    def __init__(self, mprx):
        super(GenericWorker, self).__init__()

        self.objectposeestimationrgb_proxy = mprx["ObjectPoseEstimationRGBProxy"]
        self.objectposeestimationrgbd_proxy = mprx["ObjectPoseEstimationRGBDProxy"]

