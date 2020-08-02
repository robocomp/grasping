# Grasping and Pose Estimation Components

There are three components for pose estimation, which are :

-  `objectPoseEstimationRGB` : Python component that uses `Segmentation-driven 6D Object Pose Estimation` neural network to estimate objects' poses from RGB image.

-   `objectPoseEstimationRGBD` : Python component that uses `PVN3D` neural network to estimate objects' poses from RGBD image.

-   `objectPoseEstimation` : Python component that combines both RGB and RGBD pose estimation (used in integration with DSR).

Moreover, there is one more component named `viriatoGraspingPyrep` that contains the grasping logic and the whole system integration without shared graph.

## ObjectPoseEstimationRGB Interface

It's the interface for `objectPoseEstimationRGB` component. It defines a single operation `getObjectPose`, that takes an RGB image in `TImage` format and returns `PoseType`, which is a sequence of `ObjectPose` type.

`ObjectPose` type contains :
- String representing object name.
- Translation in x, y and z as floats.
- Quaternions in x, y, z and w as floats.

Here is `ObjectPoseEstimationRGB` interface :

```
import "CameraRGBDSimple.idsl";
module RoboCompObjectPoseEstimationRGB
{
    struct ObjectPose
    {
        string objectname;
        float x;
        float y;
        float z;
        float qx;
        float qy;
        float qz;
        float qw;
    };

    sequence<ObjectPose> PoseType;

    interface ObjectPoseEstimationRGB
    {
        PoseType getObjectPose(RoboCompCameraRGBDSimple::TImage image);
    };
};
```

## ObjectPoseEstimationRGBD Interface

It's the interface for `objectPoseEstimationRGBD` and `objectPoseEstimation` components. It also defines a single operation `getObjectPose`, that takes an RGB image in `TImage` format and a depth image in `TDepth` format. Then it returns `PoseType`, which is a sequence of `ObjectPose` type.

The format of `TImage`, `ObjectPose` and `PoseType` are same as `ObjectPoseEstimationRGB` interface.

Here is `ObjectPoseEstimationRGBD` interface :

```
import "CameraRGBDSimple.idsl";
module RoboCompObjectPoseEstimationRGBD
{
    struct ObjectPose
    {
        string objectname;
        float x;
        float y;
        float z;
        float qx;
        float qy;
        float qz;
        float qw;
    };

    sequence<ObjectPose> PoseType;

    interface ObjectPoseEstimationRGBD
    {
        PoseType getObjectPose(RoboCompCameraRGBDSimple::TImage image, RoboCompCameraRGBDSimple::TDepth depth);
    };
};
```
