import sys
sys.path.append("../../segmentation-based-pose")

from api import *

import os
import numpy as np
from skimage.io import imsave
from math import tan, atan, radians, degrees

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

if not os.path.isdir("./output/"):
    os.mkdir("./output/")

# launch scene file
SCENE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenes/primitive_scene.ttt')
pr = PyRep()
pr.launch(SCENE_FILE, headless=True)
pr.start()

pr.step()

# define vision sensor
camera = VisionSensor("cam")

print("Reading vision sensor RGB signal ...")
# read vision sensor RGB image
img = camera.capture_rgb()
img = np.uint8(img * 255.0)

print("Saving output image ...")
# save sensor image
imsave("output/sim_out.png", img)

print("Getting vision sensor intrinsics ...")
# get vision sensor parameters
cam_res = camera.get_resolution()
cam_per_angle = camera.get_perspective_angle()
ratio = cam_res[0]/cam_res[1]
cam_angle_x = 0.0
cam_angle_y = 0.0
if (ratio > 1):
    cam_angle_x = cam_per_angle
    cam_angle_y = 2 * degrees(atan(tan(radians(cam_per_angle / 2)) / ratio))
else:
    cam_angle_x = 2 * degrees(atan(tan(radians(cam_per_angle / 2)) / ratio))
    cam_angle_y = cam_per_angle
# get vision sensor intrinsic matrix
cam_focal_x = float(cam_res[0] / 2.0) / tan(radians(cam_angle_x / 2.0))
cam_focal_y = float(cam_res[1] / 2.0) / tan(radians(cam_angle_y / 2.0))
intrinsics = np.array([[cam_focal_x, 0.00000000e+00, float(cam_res[0]/2.0)],
                        [0.00000000e+00, cam_focal_y, float(cam_res[1]/2.0)],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

print("Getting visual poses ...")
# classes names for ycb dataset
class_names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                            '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                            '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                            '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
# point cloud vertices of ycb models
vertices = np.load('configs/YCB-Video/YCB_vertex.npy')
# configure network
model = configure_network()
# run inference
pred_pose = get_pose(model, img, class_names, intrinsics, vertices)
# log results
print("Predicted Poses :")
print(pred_pose)

# stop simulation
pr.stop()
pr.shutdown()
