# Python RGBD Pose Estimation Component

`objectPoseEstimationRGBD` component is a Python component that performs object pose estimation on household objects for precise grasping and manipulation. This component contains an interface method `getObjectPose`, which can be directly called to get the object poses of a certain RGBD image.

## Installation

1)  Install dependencies (refer to the original [README](https://github.com/DarkGeekMS/PVN3D/blob/master/README.md)).

2)  Setup necessary folders and PointNet++ :
```bash
./setup.sh
```

3)  Download the following :
    -   [ResNet weights](https://drive.google.com/file/d/1ruEeH50E3oq7G93B8MYqs9tHo-0Nqbgw/view?usp=sharing) into `src/dnn/pvn3d/lib/ResNet_pretrained_mdl/`.
    -   [PVN3D weights](https://drive.google.com/file/d/1iLxCLve1ID8Uz_ooyd_pZMP4JXtoT1pi/view?usp=sharing) into `src/dnn/pvn3d/assets/checkpoints/`.
    -   [YCB-Videos Models](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view) into `src/dnn/pvn3d/dataset/YCB_Video_Dataset/` (for visualization).

## Configuration parameters

Like any other component, *objectPoseEstimationRGBD* needs a configuration file to start. In `etc/config`, you can change the ports and other parameters in the configuration file, according to your setting.

## Starting the component

To run `objectPoseEstimationRGBD` component, navigate to the component directory :
```bash
cd <objectPoseEstimationRGBD's path> 
```

Then compile the component :
```bash
cmake .
make
```

Then run the component :
```bash
python3 src/objectPoseEstimationRGBD.py etc/config
```
