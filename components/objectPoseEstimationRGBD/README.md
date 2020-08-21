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

### Parameters list

-   `ObjectPoseEstimationRGBD.Endpoints` : TCP port number to run `objectPoseEstimationRGBD` endpoints.
-   `weights_file` : path to PVN3D pretrained weights file.
-   Ice parameters.

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

__Note :__ Visualizations of the DNN-estimated poses will be saved to `./src/dnn/pvn3d/assets/eval_results/` directory.

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
