import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import sys
import time
import numpy as np
import scipy
import cv2
import glob
import random
import math
from pyquaternion import Quaternion

####################################################################################################################
#                                               Visualization Utilities                                            #
####################################################################################################################
def get_class_colors(class_id):
    # set colors for each class
    colordict = {'gray': [128, 128, 128], 'silver': [192, 192, 192], 'black': [0, 0, 0],
                 'maroon': [128, 0, 0], 'red': [255, 0, 0], 'purple': [128, 0, 128], 'fuchsia': [255, 0, 255],
                 'green': [0, 128, 0],
                 'lime': [0, 255, 0], 'olive': [128, 128, 0], 'yellow': [255, 255, 0], 'navy': [0, 0, 128],
                 'blue': [0, 0, 255],
                 'teal': [0, 128, 128], 'aqua': [0, 255, 255], 'orange': [255, 165, 0], 'indianred': [205, 92, 92],
                 'lightcoral': [240, 128, 128], 'salmon': [250, 128, 114], 'darksalmon': [233, 150, 122],
                 'lightsalmon': [255, 160, 122], 'crimson': [220, 20, 60], 'firebrick': [178, 34, 34],
                 'darkred': [139, 0, 0],
                 'pink': [255, 192, 203], 'lightpink': [255, 182, 193], 'hotpink': [255, 105, 180],
                 'deeppink': [255, 20, 147],
                 'mediumvioletred': [199, 21, 133], 'palevioletred': [219, 112, 147], 'coral': [255, 127, 80],
                 'tomato': [255, 99, 71], 'orangered': [255, 69, 0], 'darkorange': [255, 140, 0], 'gold': [255, 215, 0],
                 'lightyellow': [255, 255, 224], 'lemonchiffon': [255, 250, 205],
                 'lightgoldenrodyellow': [250, 250, 210],
                 'papayawhip': [255, 239, 213], 'moccasin': [255, 228, 181], 'peachpuff': [255, 218, 185],
                 'palegoldenrod': [238, 232, 170], 'khaki': [240, 230, 140], 'darkkhaki': [189, 183, 107],
                 'lavender': [230, 230, 250], 'thistle': [216, 191, 216], 'plum': [221, 160, 221],
                 'violet': [238, 130, 238],
                 'orchid': [218, 112, 214], 'magenta': [255, 0, 255], 'mediumorchid': [186, 85, 211],
                 'mediumpurple': [147, 112, 219], 'blueviolet': [138, 43, 226], 'darkviolet': [148, 0, 211],
                 'darkorchid': [153, 50, 204], 'darkmagenta': [139, 0, 139], 'indigo': [75, 0, 130],
                 'slateblue': [106, 90, 205],
                 'darkslateblue': [72, 61, 139], 'mediumslateblue': [123, 104, 238], 'greenyellow': [173, 255, 47],
                 'chartreuse': [127, 255, 0], 'lawngreen': [124, 252, 0], 'limegreen': [50, 205, 50],
                 'palegreen': [152, 251, 152],
                 'lightgreen': [144, 238, 144], 'mediumspringgreen': [0, 250, 154], 'springgreen': [0, 255, 127],
                 'mediumseagreen': [60, 179, 113], 'seagreen': [46, 139, 87], 'forestgreen': [34, 139, 34],
                 'darkgreen': [0, 100, 0], 'yellowgreen': [154, 205, 50], 'olivedrab': [107, 142, 35],
                 'darkolivegreen': [85, 107, 47], 'mediumaquamarine': [102, 205, 170], 'darkseagreen': [143, 188, 143],
                 'lightseagreen': [32, 178, 170], 'darkcyan': [0, 139, 139], 'cyan': [0, 255, 255],
                 'lightcyan': [224, 255, 255],
                 'paleturquoise': [175, 238, 238], 'aquamarine': [127, 255, 212], 'turquoise': [64, 224, 208],
                 'mediumturquoise': [72, 209, 204], 'darkturquoise': [0, 206, 209], 'cadetblue': [95, 158, 160],
                 'steelblue': [70, 130, 180], 'lightsteelblue': [176, 196, 222], 'powderblue': [176, 224, 230],
                 'lightblue': [173, 216, 230], 'skyblue': [135, 206, 235], 'lightskyblue': [135, 206, 250],
                 'deepskyblue': [0, 191, 255], 'dodgerblue': [30, 144, 255], 'cornflowerblue': [100, 149, 237],
                 'royalblue': [65, 105, 225], 'mediumblue': [0, 0, 205], 'darkblue': [0, 0, 139],
                 'midnightblue': [25, 25, 112],
                 'cornsilk': [255, 248, 220], 'blanchedalmond': [255, 235, 205], 'bisque': [255, 228, 196],
                 'navajowhite': [255, 222, 173], 'wheat': [245, 222, 179], 'burlywood': [222, 184, 135],
                 'tan': [210, 180, 140],
                 'rosybrown': [188, 143, 143], 'sandybrown': [244, 164, 96], 'goldenrod': [218, 165, 32],
                 'darkgoldenrod': [184, 134, 11], 'peru': [205, 133, 63], 'chocolate': [210, 105, 30],
                 'saddlebrown': [139, 69, 19],
                 'sienna': [160, 82, 45], 'brown': [165, 42, 42], 'snow': [255, 250, 250], 'honeydew': [240, 255, 240],
                 'mintcream': [245, 255, 250], 'azure': [240, 255, 255], 'aliceblue': [240, 248, 255],
                 'ghostwhite': [248, 248, 255], 'whitesmoke': [245, 245, 245], 'seashell': [255, 245, 238],
                 'beige': [245, 245, 220], 'oldlace': [253, 245, 230], 'floralwhite': [255, 250, 240],
                 'ivory': [255, 255, 240],
                 'antiquewhite': [250, 235, 215], 'linen': [250, 240, 230], 'lavenderblush': [255, 240, 245],
                 'mistyrose': [255, 228, 225], 'gainsboro': [220, 220, 220], 'lightgrey': [211, 211, 211],
                 'darkgray': [169, 169, 169], 'dimgray': [105, 105, 105], 'lightslategray': [119, 136, 153],
                 'slategray': [112, 128, 144], 'darkslategray': [47, 79, 79], 'white': [255, 255, 255]}

    colornames = list(colordict.keys())
    assert (class_id < len(colornames))

    r, g, b = colordict[colornames[class_id]]

    return b, g, r  # for OpenCV

def visual_img(img, folder='temp',name="out.png"):
    # visualize an RGB image
    scipy.misc.imsave(os.path.join(folder,name),img)

def visual_kp_in_img(img, kp, size=4, folder='temp', name="kp_in_img_out.png"):
    # visualize keypoints matrix on an RGB image
    # kp shape: obj X num_kp X 2
    for obj_id, obj in enumerate(kp):
        b, g, r = get_class_colors(obj_id)
        for xy in obj:
            temp_x = int(xy[0]*img.shape[1])
            temp_y = int(xy[1]*img.shape[0])
            for i in range(temp_x-size, temp_x+size):
                if i<0 or i > img.shape[1] -1 :continue
                for j in range(temp_y-size, temp_y+size):
                    if j<0 or j> img.shape[0] -1 :continue
                    img[j][i][0] = r
                    img[j][i][1] = g
                    img[j][i][2] = b
    scipy.misc.imsave(os.path.join(folder, name), img)

def visualize_predictions(predPose, image, vertex, intrinsics):
    # visualize predicted poses on RGB image
    height, width, _ = image.shape
    confImg = np.copy(image)
    maskImg = np.zeros((height,width), np.uint8)
    contourImg = np.copy(image)
    for p in predPose:
        outid, rt, conf, puv, pxyz, opoint, clsid, partid, cx, cy, layerId = p

        # show surface reprojection
        maskImg.fill(0)
        if True:
            # if False:
            vp = vertices_reprojection(vertex[outid][:], rt, intrinsics)
            for p in vp:
                if p[0] != p[0] or p[1] != p[1]:  # check nan
                    continue
                maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 1, 255, -1)
                confImg = cv2.circle(confImg, (int(p[0]), int(p[1])), 1, get_class_colors(outid), -1, cv2.LINE_AA)

        # fill the holes
        kernel = np.ones((5,5), np.uint8)
        maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
        # find contour
        contours, _ = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        contourImg = cv2.drawContours(contourImg, contours, -1, (255, 255, 255), 4, cv2.LINE_AA) # border
        contourImg = cv2.drawContours(contourImg, contours, -1, get_class_colors(outid), 2, cv2.LINE_AA)

    return contourImg


####################################################################################################################
#                                               Geometry Utilities                                                 #
####################################################################################################################
def transform_points(points_3d, mat):
    # transform 3D points with an RT matrix
    rot = np.matmul(mat[:3, :3], points_3d.transpose())
    return rot.transpose() + mat[:3, 3]

def matrix2quaternion(m):
    # get quaternions from rotation matrix
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def vertices_reprojection(vertices, rt, k):
    # project a vertex to pixel space
    p = np.matmul(k, np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def transform_pred_pose(pred_dir, object_names, transformations):
    # transform predicted poses to pixel space
    objNameList = [f for f in os.listdir(pred_dir) if os.path.isdir(pred_dir + '/' + f)]
    objNameList.sort()
    for objName in objNameList:
        objId = object_names.index(objName.lower())
        obj_dir = pred_dir + '/' + objName
        filelist = [f for f in os.listdir(obj_dir) if f.endswith('.txt')]
        for f in filelist:
            f = obj_dir + '/' + f
            pred_rt = np.loadtxt(f)
            pred_rt = np.matmul(pred_rt, transformations[objId])
            np.savetxt(f, pred_rt)

def get_bbox(label):
    # get bounding boxex for a specific label
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def iou(gt_box, est_box):
    # compute intersection over union for two boxes
    xA = max(gt_box[0], est_box[0])
    yA = max(gt_box[1], est_box[1])
    xB = min(gt_box[2], est_box[2])
    yB = min(gt_box[3], est_box[3])
    if xB <= xA or yB <= yA:
        return 0.0
    interArea = (xB - xA) * (yB - yA)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    boxBArea = (est_box[2] - est_box[0]) * (est_box[3] - est_box[1])
    # compute the intersection over union by dividing the intersection
    # area by (prediction + ground-truth areas - the interesection area)
    return interArea / float(boxAArea + boxBArea - interArea)


####################################################################################################################
#                                               Inference Utilities                                                #
####################################################################################################################
def do_detect(model, rawimg, intrinsics, bestCnt, conf_thresh, use_gpu=False):
    # perform inference on a trained model
    model.eval()
    t0 = time.time()

    height, width, _ = rawimg.shape

    # scale
    img = cv2.resize(rawimg, (model.width, model.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0) # B * 3 * W * H

    t1 = time.time()
    if use_gpu:
        img = img.cuda()
    img = Variable(img)
    t2 = time.time()

    out_preds = model(img)

    t3 = time.time()

    predPose = fusion(out_preds, width, height, intrinsics, conf_thresh, 0, bestCnt)

    t4 = time.time()

    print('-----------------------------------')
    print(' image to tensor : %f' % (t1 - t0))
    if use_gpu:
        print('  tensor to cuda : %f' % (t2 - t1))
    print('         predict : %f' % (t3 - t2))
    print('          fusion : %f' % (t4 - t3))
    print('           total : %f' % (t4 - t0))
    print('-----------------------------------')

    return predPose

def fusion(output, width, height, intrinsics, conf_thresh, batchIdx, bestCnt):
    # perform points of interest and segmentation fusion using RANSAC-based PnP
    layerCnt = len(output)
    assert(layerCnt == 2)

    cls_confs = output[0][0][batchIdx]
    cls_ids = output[0][1][batchIdx]
    predx = output[1][0][batchIdx]
    predy = output[1][1][batchIdx]
    det_confs = output[1][2][batchIdx]
    keypoints = output[1][3]

    nH, nW, nV = predx.shape
    nC = cls_ids.max() + 1

    outPred = []

    mx = predx.mean(axis=2) # average x positions
    my = predy.mean(axis=2) # average y positions
    mdConf = det_confs.mean(axis=2) # average 2D confidences
    for cidx in range(nC): # loop for every class
        # skip background
        if cidx == 0:
            continue
        foremask = (cls_ids == cidx)
        cidx -= 1

        foreCnt = foremask.sum()
        if foreCnt < 1:
            continue

        xs = predx[foremask]
        ys = predy[foremask]
        ds = det_confs[foremask]
        cs = cls_confs[foremask]
        centerxys = np.concatenate((mx[foremask].reshape(-1,1), my[foremask].reshape(-1,1)), 1)

        # choose the item with maximum detection confidence
        # actually, this will choose only one object instance for each type, this is true for OccludedLINEMOD and YCB-Video dataset
        maxIdx = np.argmax(mdConf[foremask])
        refxys = centerxys[maxIdx].reshape(1,-1).repeat(foreCnt, axis=0)
        selected = (np.linalg.norm(centerxys - refxys, axis=1) < 0.2)

        xsi = xs[selected] * width
        ysi = ys[selected] * height
        dsi = ds[selected]
        csi = cs[selected]  # confidence of selected points

        if csi.mean() < conf_thresh: # valid classification probability
            continue

        gridCnt = len(xsi)
        assert(gridCnt > 0)

        # choose best N count to define predicted keypoints
        # here N = bestCnt (default = 10)
        p2d = None
        p3d = None
        candiBestCnt = min(gridCnt, bestCnt)
        for i in range(candiBestCnt):
            bestGrids = dsi.argmax(axis=0)
            validmask = (dsi[bestGrids, list(range(nV))] > 0.5)
            xsb = xsi[bestGrids, list(range(nV))][validmask]
            ysb = ysi[bestGrids, list(range(nV))][validmask]
            t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
            t3d = keypoints[cidx][validmask]
            if p2d is None:
                p2d = t2d
                p3d = t3d
            else:
                p2d = np.concatenate((p2d, t2d), 0)
                p3d = np.concatenate((p3d, t3d), 0)
            dsi[bestGrids, list(range(nV))] = 0

        if len(p3d) < 6:
            continue

        retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)

        if not retval:
            continue

        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
        rt = np.concatenate((R, T), 1)

        outPred.append([cidx, rt, 1, None, None, None, [cidx], -1, [0], [0], None])

    return outPred


####################################################################################################################
#                                               Metrics Utilities                                                  #
####################################################################################################################
class FocalLoss(nn.Module):
    """
    Focal loss for dynamically-weighted cross entropy
    """
    def __init__(self, alpha=1, gamma=2, weights=None, reduce=True):
        super(FocalLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduce=False)
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = self.criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss

class meters:
    """
    save results and calculate average automatically
    """
    def __init__(self):
        self.value = 0.0000
        self.counter = 0
        self._reset()

    def update(self, tmp):
        self.value = (self.counter * self.value + tmp) /(self.counter+1)
        self.counter += 1

    def _reset(self):
        self.value = 0.0000
        self.counter = 0

def add_err(gt_pose, est_pose, model):
    """
    Compute average translation error of point cloud points
    Arguments:
    gt_pose  : (np.array) [4 x 4] pose matrix
    est_pose : (np.array) [4 x 4] pose matrix
    model    : (np.array) [N x 3] model 3d vertices
    """
    v_A = transform_points(model, gt_pose)
    v_B = transform_points(model, est_pose)
    v_A = np.array([x for x in v_A])
    v_B = np.array([x for x in v_B])
    return np.mean(np.linalg.norm(v_A - v_B, axis=1))

def adds_err(gt_pose, est_pose, model, sample_num=100):
    """
    Compute average translation error of point cloud points (for multiple samples)
    Arguments:
    gt_pose    : (np.array) [4 x 4] pose matrix
    est_pose   : (np.array) [4 x 4] pose matrix
    model      : (np.array) [N x 3] model 3d vertices
    sample_num : (integer) number of samples
    """
    error = []
    v_A = transform_points(model, gt_pose)
    v_B = transform_points(model, est_pose)
    for idx_A, perv_A in enumerate(v_A):
        if idx_A > sample_num:
            break
        min_error_perv_A = 10000.0
        for idx_B, perv_B in enumerate(v_B):
            if idx_B > sample_num:
                break
            if np.linalg.norm(perv_A - perv_B)<min_error_perv_A:
                min_error_perv_A = np.linalg.norm(perv_A - perv_B)
        error.append(min_error_perv_A)
    return np.mean(error)

def rot_error(gt_pose, est_pose):
    """
    Compute rotation error
    Arguments:
    gt_pose  : (np.array) [4 x 4] pose matrix
    est_pose : (np.array) [4 x 4] pose matrix
    """
    gt_quat = Quaternion(matrix2quaternion(gt_pose[:3, :3]))
    est_quat = Quaternion(matrix2quaternion(est_pose[:3, :3]))
    return np.abs((gt_quat * est_quat.inverse).degrees)

def trans_error(gt_pose, est_pose):
    """
    Compute translation error
    Arguments:
    gt_pose  : (np.array) [4 x 4] pose matrix
    est_pose : (np.array) [4 x 4] pose matrix
    """
    trans_err_norm = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    trans_err_single = np.abs(gt_pose[:3, 3] - est_pose[:3, 3])
    return trans_err_norm, trans_err_single

def projection_error_2d(gt_pose, est_pose, model, cam):
    """
    Compute 2d projection error
    Arguments:
    gt_pose  : (np.array) [4 x 4] pose matrix
    est_pose : (np.array) [4 x 4] pose matrix
    model    : (np.array) [N x 3] model 3d vertices
    cam      : (np.array) [3 x 3] camera matrix
    """
    gt_pose = gt_pose[:3]
    est_pose = est_pose[:3]
    model = np.concatenate((model, np.ones((model.shape[0], 1))), axis=1)
    gt_2d = np.matmul(np.matmul(cam, gt_pose), model.T)
    est_2d = np.matmul(np.matmul(cam, est_pose), model.T)
    gt_2d /= gt_2d[2, :]
    est_2d /= est_2d[2, :]
    gt_2d = gt_2d[:2, :].T
    est_2d = est_2d[:2, :].T
    return np.mean(np.linalg.norm(gt_2d - est_2d, axis=1))


####################################################################################################################
#                                               Miscellaneous Utilities                                            #
####################################################################################################################
def convert2cpu(gpu_matrix):
    # tensor to cpu (float)
    return torch.FloatTensor(gpu_matrix.shape).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    # tensor tp cpu (long)
    return torch.LongTensor(gpu_matrix.shape).copy_(gpu_matrix)

def read_data_cfg(datacfg):
    # read data config file
    # set gpu ids and number of cpu workers
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()
    # parse config lines
    for line in lines:
        line = line.strip()
        if len(line) > 0 and line[0] != '#' and '=' in line:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            options[key] = value
    return options

def save_predictions(imgBaseName, predPose, object_names, outpath):
    # save predicted poses to output file
    for p in predPose:
        id, rt, conf, puv, pxyz, opoint, clsid, partid, cx, cy, layerId = p
        path = outpath + '/' + object_names[int(id)] + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(path + imgBaseName + '.txt', rt)

def get_img_list_from(folder_path):
    # get images list from a specific folder
    file_list = []
    for path in glob.glob(folder_path+"/*"):
        if "jpg" in path or "png" in path:
            file_list.append(path)
    return file_list

def pnz(matrix):
    # return all non-zero elements in a matrix
    return matrix[np.where(matrix != 0)]

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl   : min erasing area
    sh   : max erasing area
    r1   : min aspect ratio
    mean : erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability=0.6, sl=0.02, sh=0.08, r1=0.5, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.shape[0] * img.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[0] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return img

        return img
