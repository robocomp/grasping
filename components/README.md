# Grasping and Pose Estimation Components

## Tests
`tests` folder contains tests for pose estimation techniques and different components.

```bash
cd tests
chmod +x run.sh
./run.sh /path/to/test/image
```

## Simple Segmentation-based Pose Estimation (V-REP/PyRep)
`simple_seg_pose_vrep` folder contains a simple component written with PyRep API to test segmentation-based pose estimation in VREP simulator.

```bash
cd simple_seg_pose_vrep
chmod +x run.sh
./run.sh
```

## Data Collector
`data_collector` folder contains the code for data collection and preparation from V-REP simulator for training data augmentation. 

Refer to `data_collector/README.md` for more information about its usage.