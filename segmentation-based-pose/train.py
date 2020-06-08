import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm
import numpy as np
import os

from ycb_dataset import YCBDataset
from segpose_net import SegPoseNet
from darknet import Darknet
from pose_2d_layer import Pose2DLayer
from pose_seg_layer import PoseSegLayer
from utils import *

## Global variables

# hyperparameters
initial_lr = 0.001
momentum = 0.9
weight_decay = 5e-4
num_epoch = 100

# data paths
ycb_root = None
ycb_data_path = None
syn_data_path = None
imageset_path = None
kp_path = None
pretrained_weights_path = None
data_cfg = None
checkpoints_dir = './models'

# dataset options
use_real_img = True
num_syn_img = 0
bg_path = None

# training options
batch_size = 32
num_workers = 4
gen_kp_gt = False
number_point = 8
modulating_factor = 1.0

# log writer
writer = SummaryWriter(logdir='./log', comment='training log')


def train(cfg_path):
    # network initialization
    data_options = read_data_cfg(cfg_path)
    model = SegPoseNet(data_options, is_train=True)

    # load pretained weights
    if pretrained_weights_path is not None:
        model.load_weights(pretrained_weights_path)
        print('Darknet weights loaded from ', pretrained_weights_path)
    
    # get input/output dimensions
    img_h = model.height
    img_w = model.width
    out_h = model.output_h
    out_w = model.output_w

    # print network graph
    model.print_network()

    model.train()

    bias_acc = meters()

    # optimizer initialization
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5*num_epoch), int(0.75*num_epoch),
                                                                 int(0.9*num_epoch)], gamma=0.1)
    
    # device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # dataset initialization
    train_dataset = YCBDataset(ycb_data_path, imageset_path, syn_data_path=syn_data_path, target_h=out_h, target_w=out_w,
                      use_real_img=use_real_img, bg_path=bg_path, num_syn_images=num_syn_img,
                                data_cfg=data_cfg, kp_path=kp_path)
    if not os.path.isfile("data/balancing_weight.pkl"):
        train_dataset.gen_balancing_weight()
    train_dataset.set_balancing_weight()
    median_balancing_weight = train_dataset.weight_cross_entropy.cuda() if use_gpu \
        else train_dataset.weight_cross_entropy

    print('training on %d images'%len(train_dataset))
    if gen_kp_gt:
        train_dataset.gen_kp_gt()

    # loss configurations
    seg_loss = nn.CrossEntropyLoss(weight=median_balancing_weight)
    pos_loss = nn.L1Loss()
    pos_loss_factor = 1.3  # 0.02 in original paper
    conf_loss = nn.L1Loss()
    conf_loss_factor = 0.8  # 0.02 in original paper

    # train/val split
    train_db, val_db = torch.utils.data.random_split(train_dataset, [len(train_dataset)-2000, 2000])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_db,
                                               batch_size=batch_size,num_workers=num_workers,
                                               shuffle=True)
    # train model
    total_step = len(train_loader)
    # loop over number of epochs
    for epoch in range(num_epoch):
        i=-1
        for images, seg_label, kp_gt_x, kp_gt_y, mask_front in tqdm(train_loader):
            i += 1
            images = images.to(device)
            seg_label = seg_label.to(device)
            kp_gt_x = kp_gt_x.to(device)
            kp_gt_y = kp_gt_y.to(device)
            mask_front = mask_front.to(device)

            # forward pass
            output = model(images)

            # segmentation
            pred_seg = output[0] # (BxOHxOW,C)
            seg_label = seg_label.view(-1)

            l_seg =seg_loss(pred_seg, seg_label)

            # regression
            mask_front = mask_front.repeat(number_point,1, 1, 1).permute(1,2,3,0).contiguous() # (B,OH,OW,NV)
            pred_x = output[1][0] * mask_front # (B,OH,OW,NV)
            pred_y = output[1][1] * mask_front
            kp_gt_x = kp_gt_x.float() * mask_front
            kp_gt_y = kp_gt_y.float() * mask_front
            l_pos = pos_loss(pred_x, kp_gt_x) + pos_loss(pred_y, kp_gt_y)

            # confidence
            conf = output[1][2] * mask_front # (B,OH,OW,NV)
            bias = torch.sqrt((pred_y-kp_gt_y)**2 + (pred_x-kp_gt_x)**2)
            conf_target = torch.exp(-modulating_factor * bias) * mask_front
            conf_target = conf_target.detach()
            l_conf = conf_loss(conf, conf_target)

            # combine all losses
            all_loss = l_seg + l_pos * pos_loss_factor + l_conf * conf_loss_factor
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # compute pixel-wise bias to measure training accuracy
                bias_acc.update(abs(pnz((pred_x - kp_gt_x).cpu()).mean()*img_w))
                print('Epoch [{}/{}], Step [{}/{}]: \n seg loss: {:.4f}, pos loss: {:.4f}, conf loss: {:.4f}, '
                      'Pixel-wise bias:{:.4f}'
                      .format(epoch + 1, num_epoch, i + 1, total_step, l_seg.item(), l_pos.item(),
                              l_conf.item(), bias_acc.value))

                writer.add_scalar('seg_loss', l_seg.item(), epoch*total_step+i)
                writer.add_scalar('pos loss', l_pos.item(), epoch*total_step+i)
                writer.add_scalar('conf_loss', l_conf.item(), epoch*total_step+i)
                writer.add_scalar('pixel_wise bias', bias_acc.value, epoch*total_step+i)

        bias_acc._reset()
        scheduler.step()
        if epoch % 5 == 1:
            model.module.save_weights(os.path.join(checkpoints_dir, f'ckpt_{epoch}.pth'))
    model.module.save_weights(os.path.join(checkpoints_dir, 'ckpt_final.pth'))
    writer.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-ds', '--dataset', type=str, help='dataset to be used for train or test', default='ycb')
    argparser.add_argument('-dsp', '--dataset_root', type=str, help='root directory of the chosen dataset')
    argparser.add_argument('-wp', '--weights_path', type=str, help='path to the pretrained weights file', default=None)

    args = argparser.parse_args()

    if not os.path.isdir('./log/'):
        os.mkdir('./log/')
    if not os.path.isdir('./models/'):
        os.mkdir('./models/')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')

    if args.dataset == 'ycb':
        # parse arguments
        ycb_root = args.dataset_root
        ycb_data_path = os.path.join(ycb_root, 'data')
        syn_data_path = os.path.join(ycb_root, 'data_syn')
        imageset_path = os.path.join(ycb_root, 'image_sets')
        kp_path = './configs/YCB-Video/YCB_bbox.npy'
        data_cfg = 'configs/data-YCB.cfg'
        pretrained_weights_path = args.weights_path
        train('./configs/data-YCB.cfg')
    else:
        print('unsupported dataset \'%s\'.' % dataset)