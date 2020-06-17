# Pose Estimation Data Collector (V-REP)

This is the code for data collection and preparation from V-REP simulator for training data augmentation.

## Dependencies

- [PyRep API](https://github.com/stepjam/PyRep) .
- [Trimesh](https://github.com/mikedh/trimesh) .

## Usage

1) Create different scenes from provided meshes in `meshes` folder. Currently, the scene must contain 4 objects (named _Shape1_, _Shape2_, _Shape3_ and _Shape4_) and a vision sensor (named _cam_). 

2) Run `get_mesh_info.py` that generates bounding boxes and vertices files in `mesh_data` folder.

```bash
python get_mesh_info.py -bbp /path/to/original/bbox/file -vp /path/to/original/vertex/file
```

3) Run `collect_sim_data.py` that collects 100 data samples from each scene in the provided directory and saves them to `sim_data` folder.

```bash
python collect_sim_data.py -sd /path/to/scenes/directory -cl /path/to/classes/json
```

Inputs :

- scenes directory.
- json file containing classes indices for each scenes.

For each output sample :

- RGB image.
- depth image.
- meta data : containing _cls_indexes_, _intrinsic_matrix_ and _poses_.

**Note :** new classes indices start from 22, as YCB-Videos dataset contains 21 classes.

4) Perform semantic segmentation on RGB images to get segmentation masks using an open-source tool like : [Image Labeling Tool](https://github.com/Slava/label-tool) .

5) Add output data to the training dataset. The following data is required :

- RGB image.
- depth image.
- segmentation mask.
- meta data `.mat` file.
- new classes to `classes.txt` .
- new paths to `image_sets` .
- new `bbox.npy` .
- new `vertex.npy` .