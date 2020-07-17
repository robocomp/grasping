#!/bin/bash
# create necessary folders and setup the network

# create necessary folders
# resnet weights folder
mkdir src/dnn/pvn3d/lib/ResNet_pretrained_mdl/
# checkpoints and outputs folder
mkdir src/dnn/pvn3d/assets/
mkdir src/dnn/pvn3d/assets/checkpoints/
mkdir src/dnn/pvn3d/assets/eval_results/
# models folder
mkdir src/dnn/pvn3d/dataset/YCB_Video_Dataset/

# setup pointnet++
cd src/dnn/
python3 setup.py build_ext
cd ../../
