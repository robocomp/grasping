#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2020 by YOUR NAME HERE
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
from PySide2 import QtWidgets, QtCore

ROBOCOMP = ''
try:
    ROBOCOMP = os.environ['ROBOCOMP']
except KeyError:
    print('$ROBOCOMP environment variable not set, using the default value /opt/robocomp')
    ROBOCOMP = '/opt/robocomp'

Ice.loadSlice("-I ./src/ --all ./src/CommonBehavior.ice")
import RoboCompCommonBehavior

Ice.loadSlice("-I ./src/ --all ./src/CameraRGBDSimple.ice")
import RoboCompCameraRGBDSimple
Ice.loadSlice("-I ./src/ --all ./src/CameraRGBDSimplePub.ice")
import RoboCompCameraRGBDSimplePub
Ice.loadSlice("-I ./src/ --all ./src/GenericBase.ice")
import RoboCompGenericBase
Ice.loadSlice("-I ./src/ --all ./src/HumanToDSR.ice")
import RoboCompHumanToDSR
Ice.loadSlice("-I ./src/ --all ./src/JoystickAdapter.ice")
import RoboCompJoystickAdapter
Ice.loadSlice("-I ./src/ --all ./src/Laser.ice")
import RoboCompLaser
Ice.loadSlice("-I ./src/ --all ./src/ObjectPoseEstimation.ice")
import RoboCompObjectPoseEstimation
Ice.loadSlice("-I ./src/ --all ./src/OmniRobot.ice")
import RoboCompOmniRobot

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

setattr(RoboCompCameraRGBDSimple, "ImgType", ImgType)

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

setattr(RoboCompCameraRGBDSimple, "DepthType", DepthType)

class People(list):
    def __init__(self, iterable=list()):
        super(People, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompHumanToDSR.Person)
        super(People, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompHumanToDSR.Person)
        super(People, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompHumanToDSR.Person)
        super(People, self).insert(index, item)

setattr(RoboCompHumanToDSR, "People", People)

class AxisList(list):
    def __init__(self, iterable=list()):
        super(AxisList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).insert(index, item)

setattr(RoboCompJoystickAdapter, "AxisList", AxisList)

class ButtonsList(list):
    def __init__(self, iterable=list()):
        super(ButtonsList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).insert(index, item)

setattr(RoboCompJoystickAdapter, "ButtonsList", ButtonsList)

class shortVector(list):
    def __init__(self, iterable=list()):
        super(shortVector, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, int)
        super(shortVector, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, int)
        super(shortVector, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, int)
        super(shortVector, self).insert(index, item)

setattr(RoboCompLaser, "shortVector", shortVector)

class TLaserData(list):
    def __init__(self, iterable=list()):
        super(TLaserData, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompLaser.TData)
        super(TLaserData, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompLaser.TData)
        super(TLaserData, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompLaser.TData)
        super(TLaserData, self).insert(index, item)

setattr(RoboCompLaser, "TLaserData", TLaserData)

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

setattr(RoboCompObjectPoseEstimation, "ImgType", ImgType)

class PoseType(list):
    def __init__(self, iterable=list()):
        super(PoseType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompObjectPoseEstimation.ObjectPose)
        super(PoseType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompObjectPoseEstimation.ObjectPose)
        super(PoseType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompObjectPoseEstimation.ObjectPose)
        super(PoseType, self).insert(index, item)

setattr(RoboCompObjectPoseEstimation, "PoseType", PoseType)


import camerargbdsimpleI
import laserI
import omnirobotI
import joystickadapterI




class GenericWorker(QtCore.QObject):

    kill = QtCore.Signal()

    def __init__(self, mprx):
        super(GenericWorker, self).__init__()

        self.objectposeestimation_proxy = mprx["ObjectPoseEstimationProxy"]
        self.camerargbdsimplepub_proxy = mprx["CameraRGBDSimplePubPub"]

        self.mutex = QtCore.QMutex(QtCore.QMutex.Recursive)
        self.Period = 30
        self.timer = QtCore.QTimer(self)


    @QtCore.Slot()
    def killYourSelf(self):
        rDebug("Killing myself")
        self.kill.emit()

    # \brief Change compute period
    # @param per Period in ms
    @QtCore.Slot(int)
    def setPeriod(self, p):
        print("Period changed", p)
        self.Period = p
        self.timer.start(self.Period)
