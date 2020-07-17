#!/bin/bash
# download PVN3D DNN weights and setup the network

# download resnet weights
mkdir src/dnn/pvn3d/lib/ResNet_pretrained_mdl/
wget "https://drive.google.com/file/d/1ruEeH50E3oq7G93B8MYqs9tHo-0Nqbgw/view?usp=sharing" -O src/dnn/pvn3d/lib/ResNet_pretrained_mdl/resnet34-333f7ec4.pth

# download pvn3d weights
mkdir src/dnn/pvn3d/assets/
mkdir src/dnn/pvn3d/assets/checkpoints/
mkdir src/dnn/pvn3d/assets/eval_results/
wget "https://drive.google.com/file/d/1iLxCLve1ID8Uz_ooyd_pZMP4JXtoT1pi/view?usp=sharing" -O src/dnn/pvn3d/assets/checkpoints/pvn3d_best.pth.tar

# download ycb models for visualization
mkdir src/dnn/pvn3d/dataset/YCB_Video_Dataset/
wget "https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view?usp=sharing" -O src/dnn/pvn3d/dataset/YCB_Video_Dataset/YCB_Video_Models.zip
unzip src/dnn/pvn3d/dataset/YCB_Video_Dataset/YCB_Video_Models.zip -d src/dnn/pvn3d/dataset/YCB_Video_Dataset/models/
rm -f src/dnn/pvn3d/dataset/YCB_Video_Dataset/YCB_Video_Models.zip

# setup pointnet++
cd src/dnn/
python3 setup.py build_ext
cd ../../
