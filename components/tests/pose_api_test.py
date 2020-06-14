import argparse
import sys
sys.path.append("./")
from api import *

import numpy as np
from skimage.io import imread
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def api_test(img_path):
    # read test image
    img = imread(img_path)
    # intrinsics of ycb dataset
    intrinsics = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
                            [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
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

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-img', '--image_path', type=str, help='path to test image')

    args = argparser.parse_args()

    # call test function
    api_test(args.image_path)
