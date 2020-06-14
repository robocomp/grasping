#!/bin/bash
# runs pose estimation api tests
# takes only one argument, which is path to test image

cd ../../segmentation-based-pose
python ../components/tests/pose_api_test.py --image_path $1