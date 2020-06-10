from utils import *
from segpose_net import SegPoseNet
from skimage.io import imsave

# Global model variables
MODEL = None # model
CLASSES = None # classes names
INTRINSICS = None # camera intrinsic matrix 
VERTICES = None # point cloud of models
BEST_CNT = None # best count
CONF_THRESH = None # confidence threshold
USE_GPU = None # device to be used
SAVE_IDX = 0 # index for output saving


def configure_network(class_names, intrinsics, vertices, bestCnt=10, conf_thresh=0.3, 
                    cfg_file='./configs/data-YCB.cfg', weights_file='./models/ckpt_final.pth',
                    use_gpu=True):
    """
    API function to configure the pose estimation network.
    Arguments:
    class_names  : inference classes names.
    intrinsics   : camera intrinsic matrix.
    vertices     : vertices of point cloud of models.
    bestCnt      : best count.
    conf_thresh  : confidence threshold.
    cfg_file     : path to config file.
    weights_file : path to pretrained weights file.
    use_gpu      : whether to use GPU or not.
    """
    # parse config data and load network
    data_options = read_data_cfg(cfg_file)
    MODEL = SegPoseNet(data_options)
    print('Building network graph ... Done!' % (weightfile))

    # print network and load weights
    m.load_weights(weights_file)
    print('Loading weights from %s... Done!' % (weightfile))

    # device selection
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    MODEL.to(device)
    print(f'Performing Inference on {device}')

    # set global variables
    CLASSES = class_names
    INTRINSICS = intrinsics
    VERTICES = vertices
    BEST_CNT = bestCnt
    CONF_THRESH = conf_thresh
    USE_GPU = (torch.cuda.is_available() and use_gpu)
    print('Configuration Done!')


def get_pose(img, save_results=True):
    """
    API function to perform pose estimation.
    Arguments:
    img          : RGB image to be used for pose estimation.
    save_results : whether to save visual results or not.
    """
    # perform pose estimation
    pred_pose = do_detect(MODEL, img, INTRINSICS, BEST_CNT, CONF_THRESH, USE_GPU)
    
    # save results
    if save_results:
        save_predictions(str(SAVE_IDX), pred_pose, CLASSES, './models') # save output poses
        visImg = visualize_predictions(pred_pose, img, VERTICES, INTRINSICS) # visualize images
        imsave('./models/' + str(SAVE_IDX) + '.jpg', visImg) # save output images

    # return predicted poses
    return pred_pose
