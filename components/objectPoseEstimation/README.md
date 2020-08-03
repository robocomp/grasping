# Python Pose Estimation Component

`objectPoseEstimation` component is a Python component that merges both `objectPoseEstimationRGB` and `objectPoseEstimationRGBD` components. This component contains an interface method `getObjectPose`, which can be directly called to get the object poses of a certain RGBD image.

Also, this component has three modes of operation :
-   RGB(0) : Inference is done through `Segmentation-driven 6D Object Pose Estimation` neural network using RGB data only.
-   RGBD(1) : Inference is done through `PVN3D` neural network using RGBD data.
-   Ensemble(2) : Inference is done through both neural networks and the results are ensembled for more precision.

## Installation

1)  Install dependencies :
-   Install CUDA9.0+
-   Install python dependencies from `requirement.txt` :
    ```bash
    pip3 install -r requirement.txt
    ```
-   Install tkinter through `sudo apt install python3-tk`.
-   Install [python-pcl](https://github.com/strawlab/python-pcl).
    -   For Ubuntu 18.04, refer to [this issue](https://github.com/strawlab/python-pcl/issues/317#issuecomment-628115649).


2)  Setup necessary folders and PointNet++ :
    ```bash
    ./setup.sh
    ```

3)  Download the following :
    -   [SegPose weights](https://drive.google.com/file/d/1N-qI5dqFVSNryZ0WwKlLn7npDkyVs_eh/view?usp=sharing) into `src/dnnlib/segpose/models/`.
    -   [ResNet weights](https://drive.google.com/file/d/1ruEeH50E3oq7G93B8MYqs9tHo-0Nqbgw/view?usp=sharing) into `src/dnnlib/pvn3d/pvn3d/lib/ResNet_pretrained_mdl/`.
    -   [PVN3D weights](https://drive.google.com/file/d/1iLxCLve1ID8Uz_ooyd_pZMP4JXtoT1pi/view?usp=sharing) into `src/dnnlib/pvn3d/pvn3d/assets/checkpoints/`.
    -   [YCB-Videos Models](https://drive.google.com/file/d/1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu/view) into `src/dnnlib/pvn3d/pvn3d/dataset/YCB_Video_Dataset/` (for visualization) _[OPTIONAL]_.

## Configuration parameters

Like any other component, *objectPoseEstimation* needs a configuration file to start. In `etc/config`, you can change the ports and other parameters in the configuration file, according to your setting.

## Starting the component

To run `objectPoseEstimation` component, navigate to the component directory :
```bash
cd <objectPoseEstimation's path> 
```

Then compile the component :
```bash
cmake .
make
```

Then run the component :
```bash
python3 src/objectPoseEstimation.py etc/config
```
