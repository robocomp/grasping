#!/bin/bash
# create necessary folders and setup PVN3D network

# PVN3D
# resnet weights folder
mkdir src/dnnlib/pvn3d/pvn3d/lib/ResNet_pretrained_mdl/
# checkpoints and outputs folder
mkdir src/dnnlib/pvn3d/pvn3d/assets/
mkdir src/dnnlib/pvn3d/pvn3d/assets/checkpoints/
mkdir src/dnnlib/pvn3d/pvn3d/assets/eval_results/
# models folder (for visualization)
mkdir src/dnnlib/pvn3d/pvn3d/dataset/YCB_Video_Dataset/

# Segmentation-driven 6D Object Pose Estimation
# checkpoints folder
mkdir src/dnnlib/segpose/models/

# setup pointnet++
cd src/dnnlib/pvn3d/
python3 setup.py build_ext
cd ../../../
