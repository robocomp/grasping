# Integration of Pose Estimation and Grasping with DSR

This file contains the full process of DSR integration with pose estimation and grasping components.

## Installation

Refer to [dsr-graph](https://github.com/robocomp/dsr-graph/#Dependencies-and-Installation) for dependencies and installation.

## Minimal Usage

Refer to [dsr-graph](https://github.com/robocomp/dsr-graph/#Basic-use-case) for minimal running components.

## Pose Estimation and Grasping Usage

_Still under development!_

## Integration Process

Refer to the [main README](https://github.com/robocomp/grasping/#System-Overview) for full description of pose estimation and grasping DSR workflow.

The process of integrating pose estimation and grasping with DSR goes as follows :

-   First, I had to finish a complete component of pose estimation, which is `objectPoseEstimation`. This component doesn't operate directly on the shared graph, however it's a separate component that is used to estimate objects' poses from RGBD to guide the grasping procedure.

-   Consequently, a component has to be developed to act as an interface between the shared graph and `objectPoseEstimation` component. That is `graspDSR` component, which is a DSR component responsible for reading the RGBD data from the shared graph, passing it to `objectPoseEstimation` component and injecting the estimated poses in the shared graph.

-   Since the final object pose can, sometimes, be hard to reach by the robot arm, `graspDSR` component has to progressively plan a set of dummy targets for the arm to follow, in order to reach the final target object. In other words, `graspDSR` component plans some keypoints on the path from current arm pose to the estimated object pose.

-   Doing so, `viriatoDSR` component passes the dummy targets to `viriatoPyrep` component, which moves the arm, using these targets, by calling the embedded Lua scripts in the arm, until the arm reaches the final target object.

-   I started developing `graspDSR` with the connection to `objectPoseEstimation`, where `graspDSR` reads all RGBD data from the shared graph and then calls `objectPoseEstimation` to get the estimated poses of the objects in the scene.

_... To be continued_

## Integration Problems

Refer to [dsr-graph issues](https://github.com/robocomp/dsr-graph/issues?q=is%3Aissue+author%3ADarkGeekMS) for problems during integration.

## Common Issues

1)  __DSR compilation requires GCC 9+, while objectPoseEstimation requires GCC 8 or older :__
    -   Install multiple C and C++ compiler versions :
        ```bash
        sudo apt install build-essential
        sudo apt -y install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9
        ```
    -   Use the `update-alternatives` tool to create list of multiple GCC and G++ compiler alternatives :
        ```bash
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
        ```
    -   Check the available C and C++ compilers list on your system and select desired version by entering relevant selection number :
        ```bash
        sudo update-alternatives --config gcc
        sudo update-alternatives --config g++
        ```
