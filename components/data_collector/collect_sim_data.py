import os
import argparse
import numpy as np
from skimage.io import imsave
from scipy.io import savemat
from math import tan, atan, radians, degrees

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape

def simulate(scene_path):
    # launch scene file
    SCENE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), scene_path)
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

    print("Reading vision sensor depth signal ...")
    # read vision sensor depth image
    depth = camera.capture_depth()

    print("Saving sensor output ...")
    # save sensor output
    imsave("data/img.png", img)
    imsave("data/depth.png", depth)

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
    
    print("Saving meta-data ...")
    # save meta-data .mat
    meta_dict = {
        'cls_indexes'      : [],
        'intrinsic_matrix' : intrinsics,
        'poses'            : []
    }
    savemat("data/meta.mat", meta_dict)

    # stop simulation
    pr.stop()
    pr.shutdown()

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-sp', '--scene_path', type=str, help='path to the scene file', default='scenes/primitive_scene.ttt')

    args = argparser.parse_args()

    if not os.path.isdir("./data/"):
        os.mkdir("./data/")

    simulate(args.scene_path)
