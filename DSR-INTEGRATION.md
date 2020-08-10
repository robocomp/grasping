# Integration of Pose Estimation and Grasping with DSR

This file contains the full process of DSR integration with pose estimation and grasping components.

## Installation

Refer to [dsr-graph](https://github.com/robocomp/dsr-graph/#Dependencies-and-Installation) for dependencies and installation.

## Minimal Usage

Refer to [dsr-graph](https://github.com/robocomp/dsr-graph/#Basic-use-case) for minimal running components.

## Pose Estimation and Grasping Integration

_Still under development!_

## Pose Estimation and Grasping Usage

_Still under development!_

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
