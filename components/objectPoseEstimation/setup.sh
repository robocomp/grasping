#!/bin/bash
# create necessary folders and setup PVN3D network

# PVN3D
# resnet weights folder
mkdir src/dnnlib/pvn3d/pvn3d/lib/ResNet_pretrained_mdl/
# pvn3d weights folder
mkdir src/dnnlib/pvn3d/pvn3d/models/
# models folder (for visualization)
mkdir src/dnnlib/pvn3d/pvn3d/dataset/YCB_Video_Dataset/

# Segmentation-driven 6D Object Pose Estimation
# checkpoints folder
mkdir src/dnnlib/segpose/models/

# Output visualization
# outputs folder
mkdir output/

# setup pointnet++
cd src/dnnlib/pvn3d/
python3 setup.py build_ext
cd ../../../
