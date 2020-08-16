# Python RGB Pose Estimation Component

`objectPoseEstimationRGB` component is a Python component that performs object pose estimation on household objects for precise grasping and manipulation. This component contains an interface method `getObjectPose`, which can be directly called to get the object poses of a certain RGB image.

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

### Parameters list

-   `ObjectPoseEstimationRGB.Endpoints` : TCP port number to run `objectPoseEstimationRGB` endpoints.
-   `config_file` : path to DNN config file.
-   `weights_file` : path to DNN pretrained weights file.
-   `vertices_file` : path to models vertices file.
-   `cam_z_offset` : calibration offset, added along camera z-axis to compensate for lost depth information (can be adjusted according to each camera setting).
-   Ice parameters.

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

Then run the component :
```bash
python3 src/objectPoseEstimationRGB.py etc/config
```

## Pose Return Description

The estimated poses are returned as a sequence of type `ObjectPose`, named `PoseType`. The data type definition goes as follows :

```
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
```

`ObjectPose` data type consists of :
-   `objectname` : string representing object name.
-   `x` : float representing translation along x-axis.
-   `y` : float representing translation along y-axis.
-   `z` : float representing translation along z-axis.
-   `qx` : float representing translation x-component of rotation quaternions.
-   `qy` : float representing translation y-component of rotation quaternions.
-   `qz` : float representing translation z-component of rotation quaternions.
-   `qw` : float representing translation w-component of rotation quaternions.

__Note :__ 

-   The estimated poses are relative to camera coordinates.

-   To project the estimated poses into camera coordinates of CoppeliaSim simulator, flip them about camera z-axis.

-   Here is a Python example of estimated poses projection into world coordinates of CoppeliaSim simulator (from `viriatoGraspingPyrep` component) :

    ```python
    def process_pose(self, obj_trans, obj_rot):
        # convert an object pose from camera frame to world frame
        # define camera pose and z-axis flip matrix
        cam_trans = self.cameras["Camera_Shoulder"]["position"]
        cam_rot_mat = R.from_quat(self.cameras["Camera_Shoulder"]["rotation"])
        z_flip = R.from_matrix(np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
        # get object position in world coordinates
        obj_trans = np.dot(cam_rot_mat.as_matrix(), np.dot(z_flip.as_matrix(), np.array(obj_trans).reshape(-1,)))
        final_trans = obj_trans + cam_trans
        # get object orientation in world coordinates
        obj_rot_mat = R.from_quat(obj_rot)
        final_rot_mat = obj_rot_mat * z_flip * cam_rot_mat
        final_rot = final_rot_mat.as_quat()
        # return final object pose in world coordinates
        final_pose = list(final_trans)
        final_pose.extend(list(final_rot))
        return final_pose
    ```
