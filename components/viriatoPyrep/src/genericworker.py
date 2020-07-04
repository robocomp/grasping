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
Ice.loadSlice("-I ./src/ --all ./src/JoystickAdapter.ice")
import RoboCompJoystickAdapter 
Ice.loadSlice("-I ./src/ --all ./src/Laser.ice")
import RoboCompLaser 
Ice.loadSlice("-I ./src/ --all ./src/LaserPub.ice")
import RoboCompLaserPub 
Ice.loadSlice("-I ./src/ --all ./src/OmniRobot.ice")
import RoboCompOmniRobot 
Ice.loadSlice("-I ./src/ --all ./src/OmniRobotPub.ice")
import RoboCompOmniRobotPub 
Ice.loadSlice("-I ./src/ --all ./src/HumanToDSRPub.ice")
import RoboCompHumanToDSRPub 

import camerargbdsimpleI 
import laserI 
import omnirobotI 
import joystickadapterI 

class GenericWorker():

    #kill = QtCore.Signal()

    def __init__(self, mprx):
        super(GenericWorker, self).__init__()

        self.camerargbdsimplepub_proxy = mprx["CameraRGBDSimplePubPub"]
        self.laserpub_proxy = mprx["LaserPubPub"]
        self.omnirobotpub_proxy = mprx["OmniRobotPubPub"]
        self.humantodsrpub_proxy = mprx["HumanToDSRPubPub"]
