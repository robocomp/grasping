# Segmentation-Based Pose Estimation

This is an implementation of [Segmentation-driven 6D Object Pose Estimation](https://arxiv.org/abs/1812.02541) paper based on the original implementation. This architecture is a single-shot segmentation-based network, in which the output points of interest and segmentation masks are fused using to RANSAC-based PnP  get the object pose.

Original implementation: https://github.com/cvlab-epfl/segmentation-driven-pose . 

## Usage

- Download [YCB videos dataset](https://rse-lab.cs.washington.edu/projects/posecnn/) or [Occluded-LINEMOD datset](https://hci.iwr.uni-heidelberg.de/vislearn/iccv2015-occlusion-challenge/) .

- Generate training or inference input list: 
`python gen_filelist.py` .

- For training:

```
python train.py
```
Provide input arguments, if required.

- For inference:
```
python test.py --gpu <boolean> --ds <dataset_name>
```

## Progress

[x] Understand, refactor and comment original repo code.

[x] Add basic code skeleton.

[x] Add training and dataset utilities.

[x] Add dataset class for YCB videos dataset.

[ ] Add dataset class for Occluded-LINEMOD datset.

[ ] Add training forward propagation to network architecture.

[ ] Add training script for network.

[ ] Train the network on given dataset.

[ ] Evaluate network performance.