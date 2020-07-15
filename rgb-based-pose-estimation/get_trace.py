import torch
import torch.nn as nn

import os
import time
import argparse
import numpy as np

from utils import *
from segpose_net import SegPoseNet

def get_trace(config_path, weights_path, out_path):
    """
    Gets the pose estimation network traced script,
    for C++ environments deployment.
    Arguments:
    config_path  : path to the network configuration file.
    weights_path : path to the network pretrained weights file.
    out_path     : path to the output traced script file.
    """
    # parse config data and load network
    data_options = read_data_cfg(config_path)
    model = SegPoseNet(data_options, False)
    print('Building network graph ... Done!')

    # print network and load weights
    model.load_weights(weights_path)
    print('Loading weights from %s... Done!' % (weights_path))

    # get model traced script
    input_sample = torch.rand(1, model.channels, model.height, model.width)
    traced_script_module = torch.jit.trace(model, input_sample)
    print('Getting network traced script ... Done!')

    # save model traced script
    traced_script_module.save(out_path)
    print('Saving network traced script ... Done!')

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-cfg', '--config_path', type=str, help='path to the network config file', default='configs/data-Custom.cfg')
    argparser.add_argument('-wp', '--weights_path', type=str, help='path to the pretrained weights file', default='models/ckpt_final.pth')
    argparser.add_argument('-op', '--out_path', type=bool, help='path to the output trace file', default='models/seg_pose_trace.pt')

    args = argparser.parse_args()

    get_trace(args.config_path, args.weights_path, args.out_path)
