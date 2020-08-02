# GSoC 2020 Grasping and Pose Estimation

Repository created for GSoC work on grasping and pose estimation. 

__Project Title:__ DNN’s for precise manipulation of household objects.

## Usage

-   Clone the repository recursively :
```bash
git clone --recurse-submodules https://github.com/robocomp/grasping.git
```

-   Follow READMEs in each sub-folder for installation and usage instructions.

## Folder Structure

-   `components` : contains all _RoboComp_ interfaces and components.

-   `data-collector` : contains the code for custom data collection using _CoppeliaSim_ and _PyRep_.

-   `rgb-based-pose-estimation` : contains the code for [Segmentation-driven 6D Object Pose Estimation](https://arxiv.org/abs/1812.02541) neural network.

-   `rgbd-based-pose-estimation` : contains the code for [PVN3D](https://arxiv.org/abs/1911.04231) neural network as a git submodule.

## System Demos

### System Overview

Our system uses PyRep API to call embedded Lua scripts in the arm for fast and precise grasping. The provided poses are estimated using DNNs through RGB(D) data collected from the shoulder camera.

### First Demo (Path Planning)

This demo verifies the arm's path planning using DNN-estimated poses. 

<div align="center">
<a href="https://www.youtube.com/watch?v=It7z-Ujf73U"><img src="https://img.youtube.com/vi/It7z-Ujf73U/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

<div align="center">
Figure(1): Video of grasping first demo.
</div><br>

<div align=center><img width="60%" height="60%" src="assets/demo1_poses.png"/></div>

<div align="center">
Figure(2): Visualization of the DNN-estimated pose in first demo.
</div><br>

### Second Demo (Grasping and Manipulation)

This demo shows the ability of the arm to grasp and manipulate a certain object out of multiple objects in the scene, using DNN-estimated poses.

<div align="center">
<a href="https://www.youtube.com/watch?v=lKk0_k8bjbY"><img src="https://img.youtube.com/vi/lKk0_k8bjbY/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

<div align="center">
Figure(3): Video of grasping second demo.
</div><br>

<div align=center><img width="60%" height="60%" src="assets/demo2_poses.png"/></div>

<div align="center">
Figure(4): Visualization of the DNN-estimated pose in second demo.
</div><br>
