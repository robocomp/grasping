# Python Pose Estimation Component

`objectPoseEstimation` component is a Python component that performs object pose estimation on household objects for precise grasping and manipulation. This components receives RGB signal from `CameraRGBDSimple` component in `ViriatoPyrep`, performs object pose estimation and then publishes the poses to `objectPoseEstimationPub`, which pushes them to the shared graph. 

**Note :** The component is under development.

## Configuration parameters
As any other component, *objectPoseEstimation* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
etc/config
```

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <objectPoseEstimation's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/objectPoseEstimation config
```
