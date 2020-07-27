# Python ViriatoPyrep For Grasping

This component is an modified copy of `viriatoPyrep` to test pose estimation and grasping with Viriato Gen3 arm (found [here](https://drive.google.com/file/d/1z7TZP6zbNzlrMwSWsogVPlnGv_FaBwRb/view?usp=sharing)) using PyRep API.

## Installation

-   Create `scenes` folder in root directory :
```bash
mkdir scenes
```

-   Download `gen3-grasp.ttt` scene file ([here](https://drive.google.com/file/d/1l5Me91K3dxAR4IpKySIRMKxOtKQmNAhH/view?usp=sharing)) and place it in `scenes` folder.

-   Install [PyRep](https://github.com/stepjam/PyRep).

-   Copy the `gen3.py` file in this directory to Pyrep directory at : `/home/xxxyour-userxxx/.local/lib/python3.x/site-packages/pyrep/robots/arms/gen3.py`.

## Configuration parameters

Like any other component, *viriatoGraspingPyrep* needs a configuration file to start. In `etc/config`, you can change the ports and other parameters in the configuration file, according to your setting.

## Starting the component

-   Compile the component :
```bash
cmake .
make
```

-   Make sure first that you have commented this two lines, in case you have them :
 ```
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

-   Start the component using the script `run.sh`. 
