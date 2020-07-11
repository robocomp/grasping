# Grasping and Pose Estimation Components

Here, we discuss the complete workflow of the components and elaborate the components interface and structure.

## Workflow Diagram

<div align=center><img width="60%" height="60%" src="assets/components_structure.png"/></div>

<div align="center">
Figure(1) : Complete schema for grasping and pose estimation workflow.
</div><br>

As shown in the figure, the components workflow goes as follows :

- `ViriatoPyrep` component streams the RGBD signal from CoppelaSim simulator using PyRep API.

- `objectPoseEstimation` component mainly consists of two components :
  - `objectPoseEstimationRGB` : receives RGB signal and performs pose estimation using `Segmentation-driven 6D Object Pose Estimation` DNN.
  - `objectPoseEstimationRGBD` : receives RGBD signal and performs pose estimation using `PVN3D` DNN.

- `objectPoseEstimation` component, then, passes the output poses to `objectPoseEstimationPub` component, which publishes the poses to the shared memory in `ViriatoDSR`.

- `Grasping` component streams the poses from the shared memory and uses it to plan a grasp on the object.

## Pose Estimation Interfaces

As discussed, there are three components for pose estimation, which are `objectPoseEstimationRGB` (Python), `objectPoseEstimationRGBD` (Python) and `objectPoseEstimationPub` (C++).

### ObjectPoseEstimationRGB Interface

It's the interface for `objectPoseEstimationRGB` component. It defines a single operation `getObjectPose`, that takes an RGB image in `TImage` format and returns `PoseType`, which is a sequence of `ObjectPose` type.

`ObjectPose` type contains :
- String representing object name.
- Translation in x, y and z as floats.
- Quaternions in x, y, z and w as floats.

Here is `ObjectPoseEstimationRGB` interface :

```
module RoboCompObjectPoseEstimationRGB
{
    exception HardwareFailedException { string what; };

    sequence<byte> ImgType;

    struct TImage
    {
        int width;
        int height;
        int depth;
        int focalx;
        int focaly;
        ImgType image;
    };

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
        PoseType getObjectPose(TImage img) throws HardwareFailedException;
    };
};
```

### ObjectPoseEstimationRGBD Interface

It's the interface for `objectPoseEstimationRGBD` component. It also defines a single operation `getObjectPose`, that takes an RGB image in `TImage` format and a depth image in `TDepth` format. Then it returns `PoseType`, which is a sequence of `ObjectPose` type.

The format of `TImage`, `ObjectPose` and `PoseType` are taken from `ObjectPoseEstimationRGB` interface.

Here is `ObjectPoseEstimationRGB` interface :

```
import "ObjectPoseEstimationRGB.idsl";

module RoboCompObjectPoseEstimationRGBD
{
    exception HardwareFailedException { string what; };

    sequence<byte> DepthType;

    struct TDepth
    {
        int width;
        int height;
        DepthType depth;
    };

    interface ObjectPoseEstimationRGBD
    {
        RoboCompObjectPoseEstimationRGB::PoseType getObjectPose(RoboCompObjectPoseEstimationRGB::TImage image, TDepth depth) throws HardwareFailedException;
    };
};
```

### ObjectPoseEstimationPub Interface

It's the interface for `objectPoseEstimationPub` component. It defines a single operation `pushObjectPose`, which simply pushes the obtained object poses into the shared memory. 

Here is `ObjectPoseEstimationPub` interface :

```
import "ObjectPoseEstimationRGB.idsl";

module RoboCompObjectPoseEstimationPub
{
  interface ObjectPoseEstimationPub
  {
    idempotent void pushObjectPose(RoboCompObjectPoseEstimationRGB::PoseType poses);
  };
};
```