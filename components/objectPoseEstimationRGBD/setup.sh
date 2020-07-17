#!/bin/bash
# download PVN3D DNN weights and setup the network

# get resnet weights
mkdir src/dnn/pvn3d/lib/ResNet_pretrained_mdl/
wget "https://drive.google.com/file/d/1ruEeH50E3oq7G93B8MYqs9tHo-0Nqbgw/view?usp=sharing" -O src/dnn/pvn3d/lib/ResNet_pretrained_mdl/resnet34-333f7ec4.pth

# get pvn3d weights
mkdir src/dnn/pvn3d/assets/
mkdir src/dnn/pvn3d/assets/checkpoints/
mkdir src/dnn/pvn3d/assets/eval_results/
wget "https://drive.google.com/file/d/1iLxCLve1ID8Uz_ooyd_pZMP4JXtoT1pi/view?usp=sharing" -O src/dnn/pvn3d/assets/checkpoints/pvn3d_best.pth.tar

# setup pointnet++
cd src/dnn/
python3 setup.py build_ext
cd ../../
