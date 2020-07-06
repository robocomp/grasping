# Python ViriatoPyrep For Grasping

This component is an edited copy of `viriatoPyrep` to test pose estimation and grasping with Viriato Gen3 arm using PyRep API.

**Note :** The component is under development.

## Installation

-   Download `gen3-grasp.ttt` scene file using `get_scene.sh`.

-   Install [PyRep](https://github.com/stepjam/PyRep).

-   Copy the `gen3.py` file in this directory to Pyrep directory at : `/home/xxxyour-userxxx/.local/lib/python3.x/site-packages/pyrep/robots/arms/gen3.py`.

## Configuration parameters

Like any other component, *objectPoseEstimation* needs a configuration file to start. In `etc/config`, you can change the ports and other parameters in the configuration file, according to your setting.

## Starting the component

-   Compile the component :
```
cmake .
make
```

-   Make sure first that you have commented this two lines, in case you have them :
 ```
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

-   Start the component copy the config file form etc to . and use the script `run.sh`. 

-   If you have a joystick, start `~/robocomp/components/robocomp-robolab/components/hardware/external_control/joystickpublish`. Check the config file to set the ranges of the axis.
