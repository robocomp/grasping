# Python RGB Pose Estimation Component

`objectPoseEstimationRGB` component is a Python component that performs object pose estimation on household objects for precise grasping and manipulation. This components receives RGB signal from `CameraRGBDSimple` component in `ViriatoPyrep`, performs object pose estimation using `Segmentation-driven 6D Object Pose Estimation` DNN and then publishes the poses to `objectPoseEstimationPub`, which pushes them to the shared graph.

The `objectPoseEstimationRGB` component continuously publishes the predicted poses into the shared graph in `ViriatoDSR` through `objectPoseEstimationPub`. However, it contains an interface method `getObjectPose`, which can be directly called to get the object poses of a certain RGB image.

## Installation

1)  Install dependencies :
```bash
pip install torch torchvision
pip install tensorboardX
pip install opencv-python
pip install opencv-contrib-python
pip install skimage tqdm pyquaternion
```

2)  Get the pretrained weights :
    -   Download [here](https://drive.google.com/file/d/1N-qI5dqFVSNryZ0WwKlLn7npDkyVs_eh/view?usp=sharing).
    -   Place it in `src/models`.

## Configuration parameters

Like any other component, *objectPoseEstimationRGB* needs a configuration file to start. In `etc/config`, you can change the ports and other parameters in the configuration file, according to your setting.

## Starting the component

To run `objectPoseEstimationRGB` component, navigate to the component directory :
```bash
cd <objectPoseEstimationRGB's path> 
```

Then compile the component :
```bash
cmake .
make
```

Then start _rcnode_ in a separate terminal (for pub/sub services) :
```bash
rcnode
``` 

Then run the component :
```bash
python3 src/objectPoseEstimationRGB.py etc/config
```
