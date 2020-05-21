import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import *
from tqdm import tqdm
import random
import numpy as np

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
        with open(opj(imageset_path, "trainval.txt"), 'r') as file:
            trainlines = file.readlines()
        # general absolute paths for the training samples
        real_train_path = [opj(self.root,x.rstrip('\n')) for x in trainlines]
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

            # get front image mask
            mask_front = ma.getmaskarray(ma.masked_not_equal(label_img, 0)).astype(int)

            # return training data
            # input  : normalized RGB image & segmentation mask
            # output : x ground truth map, y ground truth map & front mask
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
