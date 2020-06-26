#!/bin/bash
# runs pose estimation api tests
# takes only one argument, which is path to test image

cd ../segmentation-based-pose
python ../tests/pose_api_test.py --dataset $1 --image_path $2 --weights_path $3