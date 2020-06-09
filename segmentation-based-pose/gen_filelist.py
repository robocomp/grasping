import os
import argparse

def collect_occluded_linemod_testlist(rootpath, outname):
    path = rootpath + 'RGB-D/rgb_noseg/'
    imgs = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    imgs.sort()
    # write sets
    allf = open(outname, 'w')
    for i in imgs:
        allf.write(path + i +'\n')

def collect_ycb_testlist(rootpath, outfile):
    testListFile = rootpath + '/image_sets/keyframe.txt'
    with open(testListFile, 'r') as file:
        testlines = file.readlines()
    with open(outfile, 'w') as file:
        for l in testlines:
            file.write(rootpath + 'data/' + l.rstrip() + '-color.png\n')

