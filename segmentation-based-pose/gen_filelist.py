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

if __name__ == '__main__':
    # parse dataset name and path arguments
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-ds', '--dataset', type=str, help='dataset name', default='linemod')
    argparser.add_argument('-pt', '--path', type=str, help='dataset path', default='/data/OcclusionChallengeICCV2015/')
    argparser.add_argument('-tf', '--testfile', type=str, help='test file path', default='./occluded-linemod-testlist.txt')

    args = argparser.parse_args()

    dataset_name = args.dataset
    dataset_path = args.path
    testfile_path = args.testfile

    if dataset_name = 'linemod':
        collect_occluded_linemod_testlist(dataset_path, testfile_path)
    elif dataset_name = 'ycb':
        collect_ycb_testlist(dataset_path, testfile_path)
    else:
        print("Invalid dataset name!")
