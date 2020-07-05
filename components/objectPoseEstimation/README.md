# Python Pose Estimation Component

`objectPoseEstimation` component is a Python component that performs object pose estimation on household objects for precise grasping and manipulation. This components receives RGB signal from `CameraRGBDSimple` component in `ViriatoPyrep`, performs object pose estimation and then publishes the poses to `objectPoseEstimationPub`, which pushes them to the shared graph.

The `objectPoseEstimation` component continuously publishes the predicted poses into the shared graph in `ViriatoDSR` through `objectPoseEstimationPub`. However, it contains an interface method `getObjectPose`, which can be directly called to get the object poses of a certain RGB image.

**Note :** The component is under development.

## Installation

1)  Install dependencies :
```bash
pip install torch torchvision
pip install tensorboardX
pip install opencv-python
pip install opencv-contrib-python
pip install skimage tqdm pyquaternion
```

2)  Download pretrained weights :
```bash
chmod +x get_weights.sh
./get_weights.sh
```

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

After editing the config file, we can run the component :
```bash
python3 src/objectPoseEstimation.py etc/config
```
