import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import os
import random
import pickle
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat

from utils import *

class YCBDataset(Dataset):
    def __init__(self, root, imageset_path, syn_data_path=None, use_real_img = True, num_syn_images=200000 ,target_h=76, target_w=76, 
                bg_path = None, kp_path="data/YCB-Video/YCB_bbox.npy", data_cfg="data/data-YCB.cfg",
                use_bg_img = True):
        self.root = root # data root directory
        data_options = read_data_cfg(data_cfg) # data config

        self.input_width = int(data_options['width']) # width of input image (==608)
        self.input_height = int(data_options['height']) # height of input image (==608)
        
        self.original_width = 640 # width of original img
        self.original_height = 480 # height of original img

        self.target_h = target_h # output feature map width
        self.target_w = target_w # output feature map width

        self.num_classes = int(data_options['classes']) # number of classes

        self.train_paths = [] # paths of training samples
        self.gen_train_list(imageset_path) # generate paths of training samples

        self.use_real_img = use_real_img # whether to use real or synthetic data from YCB

        self.syn_data_path = syn_data_path # path to synthetic data
        self.syn_range = 80000 # number of synthetic data in YCB (80000 sequentially indexed)

        self.syn_bg_image_paths = get_img_list_from(bg_path) if bg_path is not None else [] # paths of background images
        self.use_bg_img = use_bg_img # whether to use background images or not
        self.num_syn_images = num_syn_images # number of synthetic images for training

        self.weight_cross_entropy = None # weight for CE with ratio between real to synthetic
        self.set_balancing_weight() # weight CE 

        # YCB classes names
        self.object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                 '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
                                 '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
                                 '024_bowl', '025_mug', '035_power_drill',
                                 '036_wood_block', '037_scissors', '040_large_marker', '051_large_clamp',
                                 '052_extra_large_clamp',
                                 '061_foam_brick']
        self.ycb_class_to_idx = {} # class name to index
        for i, item in enumerate(self.object_names_ycbvideo):
            self.ycb_class_to_idx[item] = i

        self.kp3d = np.load(kp_path) # camera intrinsic matrices
        self.n_kp = 8 # number of intrinsic properties

    def gen_train_list(self, imageset_path, out_pkl="data/real_train_path.pkl"):
        # read train/validation list
        with open(os.path.join(imageset_path, "trainval.txt"), 'r') as file:
            trainlines = file.readlines()
        # general absolute paths for the training samples
        real_train_path = [os.path.join(self.root,x.rstrip('\n')) for x in trainlines]
        with open(out_pkl, 'wb') as f:
            pickle.dump(real_train_path, f)
        self.train_paths = real_train_path

    def set_balancing_weight(self, save_pkl="data/balancing_weight.pkl"):
        # read file with data frequencies
        print("Loading weight from file ", save_pkl)
        with open(save_pkl, 'rb') as f:
            frequencies = pickle.load(f)
        real_frequency = frequencies['real']
        syn_frequency = frequencies['syn']
        # generate weights
        combined_frequency = self.num_syn_images * syn_frequency + len(self.train_paths) * real_frequency
        median_frequency = np.median(combined_frequency)
        weight = [median_frequency/x for x in combined_frequency]
        # set CE weight to be used during training
        self.weight_cross_entropy =  torch.from_numpy(np.array(weight)).float()

    def gen_balancing_weight(self, save_pkl="data/balancing_weight.pkl"):
        # get pixel-wise balancing weight for cross entropy loss
        pixels_per_img = (self.target_h * self.target_w)
        real_frequency = [0 for x in range(self.num_classes)]

        # get weights of real images
        print("collect weight for real images")
        for prefix in tqdm(self.train_paths):
            label_img = cv2.imread(prefix + "-label.png")[: , : , 0]
            label_img = cv2.resize(label_img, (self.target_h, self.target_w), interpolation=cv2.INTER_NEAREST)
            labels_per_img = np.unique(label_img)
            for img_id in labels_per_img:
                if len(np.where(label_img==img_id)) <1:
                    real_frequency[img_id] += 0
                else:
                    real_frequency[img_id] += len(np.where(label_img==img_id)[0]) / pixels_per_img
        real_frequency = np.array(real_frequency)
        real_frequency/=len(self.train_paths)

        # get weights of synthetic images
        print("collect weights for syn images")
        syn_frequency = [0 for x in range(self.num_classes)]
        prefix = self.syn_data_path
        for id in tqdm(range(self.syn_range - 1)):
            item = os.path.join(prefix, "%06d"%id)
            seg_img = cv2.imread(item + "-label.png")
            seg_img = cv2.resize(seg_img, (self.target_h, self.target_w), interpolation=cv2.INTER_NEAREST)
            labels_per_img = np.unique(seg_img)
            for img_id in labels_per_img:
                if len(np.where(seg_img==img_id)) <1:
                    syn_frequency[img_id] += 0
                else:
                    syn_frequency[img_id] += len(np.where(seg_img==img_id)[0]) / pixels_per_img
        
        # normalize frequencies and write them to output file
        syn_frequency = np.array(syn_frequency)
        syn_frequency/=self.syn_range
        frequencies = {'real':real_frequency, 'syn':syn_frequency}
        with open(save_pkl, 'wb') as f:
            pickle.dump(frequencies, f)

    def gen_kp_gt_for_item(self, item):
        # item is a path prefix
        out_pkl = item + '-bb8_2d.pkl'
        # read meta data
        meta = loadmat(item + '-meta.mat')
        intrinsic = meta['intrinsic_matrix']
        poses = meta['poses'].transpose(2, 0, 1)
        cls_idxs = meta['cls_indexes'] - 1
        cls_idxs = cls_idxs.squeeze()
        # get 2d kp matrix
        kp_2d = np.zeros((len(cls_idxs), self.n_kp, 2))
        for idx, pose in enumerate(poses):
            vertex = self.kp3d[int(cls_idxs[idx])].squeeze()
            kp_2d[idx] = vertices_reprojection(vertex, pose, intrinsic)
        # normalize kp matrix
        kp_2d[:, :, 0] /= self.original_width
        kp_2d[:, :, 1] /= self.original_height
        # dump into output file to be used later in training
        with open(out_pkl, 'wb') as f:
            pickle.dump(kp_2d, f)

    def gen_kp_gt(self, for_syn = True, for_real = True):
        if for_real:
            # generate kp for all real images
            print("generate and save kp gt for real images.")
            for item in tqdm(self.train_paths):
                self.gen_kp_gt_for_item(item)
        if for_syn:
            # generate kp for synthetic images
            print("generate and save kp gt for synthetic images.")
            syn_prefix = self.syn_data_path
            for id in tqdm(range(self.syn_range)):
                item = os.path.join(syn_prefix, "%06d" % id)
                self.gen_kp_gt_for_item(item)

    def gen_synthetic(self):
        # check for background paths
        if len(self.syn_bg_image_paths)<1 :
            print("you need to give bg images folder!")

        # generate a synthetic image on the fly
        prefix = self.syn_data_path
        id = random.randint(0, self.syn_range-1)
        item = os.path.join(prefix, "%06d"%id)
        raw = cv2.imread(item + "-color.png")
        img = cv2.resize(raw, (self.input_height, self.input_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get segmentation gt
        seg_img = cv2.imread(item + "-label.png")
        seg_img = cv2.resize(seg_img, (self.input_height, self.input_width), interpolation=cv2.INTER_NEAREST)
        mask_front = ma.getmaskarray(ma.masked_not_equal(seg_img, 0)).astype(int)
        mask_back = ma.getmaskarray(ma.masked_equal(seg_img, 0)).astype(int)

        # random erase some parts to make the network robust to occlusions
        random_erasing = RandomErasing(sl=0.01,sh=0.1)
        img = random_erasing(img)

        # get bg image and combine them together
        back_img_path = random.choice(self.syn_bg_image_paths)
        bg_raw = cv2.imread(back_img_path)
        bg_img = cv2.resize(bg_raw, (self.input_height, self.input_width))
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        if self.use_bg_img:
            combined_img = bg_img * mask_back + img * mask_front
        else:
            combined_img = img * mask_front
        color_augmentation = transforms.ColorJitter(0.02, 0.02, 0.02, 0.05)
        combined_img = Image.fromarray(combined_img.astype('uint8')).convert('RGB')
        combined_img = color_augmentation(combined_img)
        combined_img = np.array(combined_img)

        # get segmentation label
        seg_label = seg_img[:, :, 0] # RGB channels are the same
        seg_label = cv2.resize(seg_label, (self.target_h, self.target_w), interpolation=cv2.INTER_NEAREST)

        # generate kp gt map of (nH, nW, nV)
        kp_gt_map_x = np.zeros((self.target_h, self.target_w, self.n_kp))
        kp_gt_map_y = np.zeros((self.target_h, self.target_w, self.n_kp))
        in_pkl = item + '-bb8_2d.pkl'

        # load class info
        meta = loadmat(item + '-meta.mat')
        class_ids = meta['cls_indexes']
        with open(in_pkl, 'rb') as f:
            bb8_2d = pickle.load(f)
        for i, cid in enumerate(class_ids):
            class_mask = np.where(seg_label == cid[0])
            kp_gt_map_x[class_mask] = bb8_2d[:,:,0][i]
            kp_gt_map_y[class_mask] = bb8_2d[:,:,1][i]

        # get image mask front (used to compute loss)
        mask_front = cv2.resize(mask_front, (self.target_h, self.target_w), interpolation=cv2.INTER_NEAREST)

        # return training data
        # input  : normalized RGB image & segmentation mask
        # output : x ground truth map, y ground truth map & mask front
        return (torch.from_numpy(combined_img.transpose(2, 0, 1)).float().div(255.0),
                torch.from_numpy(seg_label).long(),
                torch.from_numpy(kp_gt_map_x).float(), torch.from_numpy(kp_gt_map_y).float(),
                torch.from_numpy(mask_front[:,:,0]).float())

    def __getitem__(self, index):
        # get a single training sample
        if not self.use_real_img:
            # use synthetic images
            return self.gen_synthetic()
        if index > len(self.train_paths) - 1:
            # generate synthetic images if index out of original data range
            return self.gen_synthetic()
        else:
            # use real images from YCB videos
            prefix = self.train_paths[index]

            # get raw image
            raw = cv2.imread(prefix + "-color.png")
            img = cv2.resize(raw, (self.input_height, self.input_width))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # load class info
            meta = loadmat(prefix + '-meta.mat')
            class_ids = meta['cls_indexes']

            # get segmentation gt, note 0 is for background
            label_img = cv2.imread(prefix + "-label.png")[: , : , 0]
            label_img = cv2.resize(label_img, (self.target_h, self.target_w), interpolation=cv2.INTER_NEAREST)

            # generate kp gt map of (nH, nW, nV)
            kp_gt_map_x = np.zeros((self.target_h, self.target_w, self.n_kp))
            kp_gt_map_y = np.zeros((self.target_h, self.target_w, self.n_kp))
            in_pkl = prefix + '-bb8_2d.pkl'
            with open(in_pkl, 'rb') as f:
                bb8_2d = pickle.load(f)
            for i, cid in enumerate(class_ids):
                class_mask = np.where(label_img == cid[0])
                kp_gt_map_x[class_mask] = bb8_2d[:,:,0][i]
                kp_gt_map_y[class_mask] = bb8_2d[:,:,1][i]

            # get image mask front (used to compute loss)
            mask_front = ma.getmaskarray(ma.masked_not_equal(label_img, 0)).astype(int)

            # return training data
            # input  : normalized RGB image & segmentation mask
            # output : x ground truth map, y ground truth map & mask front
            return (torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0),
                    torch.from_numpy(label_img).long(),
                    torch.from_numpy(kp_gt_map_x).float(), torch.from_numpy(kp_gt_map_y).float(),
                    torch.from_numpy(mask_front).float())

    def __len__(self):
        # get dataset length
        if self.use_real_img:
            return len(self.train_paths)+self.num_syn_images
        else:
            return self.num_syn_images
