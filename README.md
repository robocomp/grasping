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
