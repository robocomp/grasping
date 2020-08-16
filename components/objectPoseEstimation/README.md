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

### Parameters list

-   `ObjectPoseEstimationRGBD.Endpoints` : TCP port number to run `objectPoseEstimation` endpoints.
-   `rgb_config_file` : path to `Segmentation-driven 6D Object Pose Estimation` config file.
-   `rgb_weights_file` : path to `Segmentation-driven 6D Object Pose Estimation` pretrained weights file.
-   `rgb_vertices_file` : path to models vertices file.
-   `rgbd_weights_file` : path to PVN3D pretrained weights file.
-   `rgb_cam_z_offset` : calibration offset, added along camera z-axis to compensate for lost depth information in case of RGB pose estimation (can be adjusted according to each camera setting).
-   `inference_mode` : an integer that defines inference mode of pose estimation, whether it's _RGB(0)_, _RGBD(1)_ or _Ensemble(2)_.
-   Ice parameters.

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

## Objects Names Mapping

`YCB-Videos dataset` objects can be found [here](https://github.com/robocomp/grasping/tree/master/data-collector/meshes/ycb). The following table shows the mapping between `objectPoseEstimation` return names and `YCB-Videos dataset` objects names.

| Return Name        |      YCB Model        |
|--------------------|-----------------------|
|  can_1             |  002_master_chef_can  |
|  box_1             |  003_cracker_box      |
|  box_2             |  004_sugar_box        |
|  can_2             |  005_tomato_soup_can  |
|  bottle_1          |  006_mustard_bottle   |
|  can_3             |  007_tuna_fish_can    |
|  box_3             |  008_pudding_box      |
|  box_4             |  009_gelatin_box      |
|  can_4             |  010_potted_meat_can  |
|  banana_1          |  011_banana           |
|  pitcher_base_1    |  019_pitcher_base     |
|  bleach_cleanser_1 |  021_bleach_cleanser  |
|  bowl_1            |  024_bowl             |
|  mug_1             |  025_mug              |
|  power_drill_1     |  035_power_drill      |
|  wood_block_1      |  036_wood_block       |
|  scissors_1        |  037_scissors         |
|  marker_1          |  040_large_marker     |
|  large_clamp_1     |  051_large_clamp      |
|  large_clamp_2     |  052_extra_large_clamp|
|  foam_brick_1      |  061_foam_brick       |
