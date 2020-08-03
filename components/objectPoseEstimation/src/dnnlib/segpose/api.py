from dnnlib.segpose.dnn.utils import *
from dnnlib.segpose.dnn.segpose_net import SegPoseNet
from skimage.io import imsave

def configure_network(cfg_file='configs/data-YCB.cfg', weights_file='models/ckpt_final.pth', use_gpu=True):
    """
    API function to configure the pose estimation network.
    Arguments:
    cfg_file     : path to config file.
    weights_file : path to pretrained weights file.
    use_gpu      : whether to use GPU or not.
    """
    # parse config data and load network
    data_options = read_data_cfg(cfg_file)
    model = SegPoseNet(data_options, False)
    print('Building network graph ... Done!')

    # print network and load weights
    model.load_weights(weights_file)
    print('Loading weights from %s... Done!' % (weights_file))

    # device selection
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    model.to(device)
    print(f'Performing Inference on {device}')

    # return model object
    return model


def get_pose(model, img, class_names, intrinsics, vertices, best_cnt=10, conf_thresh=0.3, 
            save_results=True, output_file = 'out', use_gpu=True):
    """
    API function to perform pose estimation.
    Arguments:
    model        : configured model object.
    img          : RGB image to be used for pose estimation.
    class_names  : inference classes names.
    intrinsics   : camera intrinsic matrix.
    vertices     : vertices of point cloud of models.
    bestCnt      : best count.
    conf_thresh  : confidence threshold.
    save_results : whether to save visual results or not.
    output_file  : name of output file
    use_gpu      : whether to use GPU or not.
    """
    # perform pose estimation
    pred_pose = do_detect(model, img, intrinsics, best_cnt, conf_thresh, use_gpu)
    
    # save results
    if save_results:
        if not os.path.isdir('output/'):
            os.mkdir('output/')
        save_predictions(str(output_file), pred_pose, class_names, 'output') # save output poses
        vis_img = visualize_predictions(pred_pose, img, vertices, intrinsics) # visualize images
        imsave('output/' + str(output_file) + '.jpg', vis_img) # save output images

    # return predicted poses
    return pred_pose
