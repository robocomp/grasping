import argparse
from utils import *
from gen_filelist import *
from segpose_net import SegPoseNet
from skimage.io import imread, imsave
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(data_cfg, weightfile, listfile, outdir, object_names, intrinsics, vertex,
                         bestCnt, conf_thresh, linemod_index=False, use_gpu=False, gpu_id='0'):
    """
    Main pose estimation testing driver,
    Used to run inference on network and save visual results.
    Arguments:
    data_cfg      : path to data config file.
    weightfile    : path to pretrained weights file.
    listfile      : path to text file with list of test images.
    outdir        : path to output directory.
    object_names  : list of object names in dataset.
    intrinsics    : intrinsic matrix of camera.
    vertex        : vertex coordinates extracted from dataset for visualization.
    bestCnt       : best count.
    conf_thresh   : confidence threshold.
    linemod_index : whether to use linemod index or not.
    use_gpu       : whether to use gpu or not.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # parse config data and load network
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options, False)

    # print network and load weights
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        m.cuda()

    # read list of test images
    with open(listfile, 'r') as file:
        imglines = file.readlines()

    # loop over all images in test list
    for idx in range(len(imglines)):
        imgfile = imglines[idx].rstrip()
        img = imread(imgfile) # read image

        dirname, filename = os.path.split(imgfile)
        baseName, _ = os.path.splitext(filename)
        if linemod_index:
            outFileName = baseName[-4:]
        else:
            dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
            outFileName = dirname+'_'+baseName

        start = time.time()
        predPose = do_detect(m, img, intrinsics, bestCnt, conf_thresh, use_gpu) # perform pose estimation
        finish = time.time()

        arch = 'CPU'
        if use_gpu:
            arch = 'GPU'
        print('%s: Predict %d objects in %f seconds (on %s).' % (imgfile, len(predPose), (finish - start), arch))
        save_predictions(outFileName, predPose, object_names, outdir) # save output poses

        # visualize predictions
        vis_start = time.time()
        visImg = visualize_predictions(predPose, img, vertex, intrinsics)
        imsave(outdir + '/' + outFileName + '.jpg', visImg)
        vis_finish = time.time()
        print('%s: Visualization in %f seconds.' % (imgfile, (vis_finish - vis_start)))


if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-gpu', '--use_gpu', type=bool, help='set to true to use gpu, otherwise use cpu', default=True)
    argparser.add_argument('-ds', '--dataset', type=str, help='dataset to be used for train or test', default='ycb')
    argparser.add_argument('-dsp', '--dataset_root', type=str, help='root directory of the chosen dataset')
    argparser.add_argument('-wp', '--weights_path', type=str, help='path to the pretrained weights file', default='models/ckpt_final.pth')

    args = argparser.parse_args()

    if args.dataset == 'linemod':
        # generate test list file for linemod
        listfile = './data/occluded-linemod-testlist.txt'
        collect_ycb_testlist(args.dataset_root, listfile)
        # intrinsics of LINEMOD dataset
        k_linemod = np.array([[572.41140, 0.0, 325.26110],
                              [0.0, 573.57043, 242.04899],
                              [0.0, 0.0, 1.0]])
        # 8 objects for LINEMOD dataset
        object_names_occlinemod = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
        vertex_linemod = np.load('./configs/Occluded-LINEMOD/LINEMOD_vertex.npy')
        test('./configs/data-LINEMOD.cfg',
                    args.weights_path, listfile,
                    './output', object_names_occlinemod, k_linemod, vertex_linemod,
                    bestCnt=10, conf_thresh=0.3, linemod_index=True, use_gpu=args.use_gpu)
        # LINEMOD visualization transforms
        rt_transforms = np.load('./configs/Occluded-LINEMOD/Transform_RT_to_OccLINEMOD_meshes.npy')
        transform_pred_pose('./output', object_names_occlinemod, rt_transforms)
    elif args.dataset == 'ycb':
        # generate test list file for linemod
        listfile = './data/ycb-video-testlist.txt'
        collect_ycb_testlist(args.dataset_root, listfile)
        # intrinsics of YCB-VIDEO dataset
        k_ycbvideo = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
                               [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        vertex_ycbvideo = np.load('./configs/YCB-Video/YCB_vertex.npy')
        test('./configs/data-YCB.cfg',
                    args.weights_path, listfile,
                    './output', object_names_ycbvideo, k_ycbvideo, vertex_ycbvideo,
                    bestCnt=10, conf_thresh=0.3, use_gpu=args.use_gpu)
    else:
        print('unsupported dataset \'%s\'.' % dataset)