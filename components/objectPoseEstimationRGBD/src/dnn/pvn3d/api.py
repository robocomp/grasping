import torch
import torch.nn as nn

import os
import cv2
import pcl
import argparse
import numpy as np
from PIL import Image
import pickle as pkl

from lib import PVN3D
from common import Config
from lib.utils.basic_utils import Basic_Utils
from datasets.ycb.ycb_dataset import YCB_Dataset
from lib.utils.sync_batchnorm import convert_model
from lib.utils.pvn3d_eval_utils import cal_frame_poses


class RGBDPoseAPI():
    r"""Interface of PVN3D network for inference on a single RGBD image.

    Parameters
    ----------
    weights_path : str
        Path to weights file of the network
    """
    def __init__(self, weights_path):
        # initialize configs and model object
        self.config = Config(dataset_name='ycb')
        self.bs_utils = Basic_Utils(self.config)
        self.model = self.define_network(weights_path)
        self.rgb = None
        self.cld = None
        self.cld_rgb_nrm = None
        self.choose = None
        self.cls_id_lst = None

    def load_checkpoint(self, model=None, optimizer=None, filename="checkpoint"):
        # load network checkpoint from weights file
        filename = "{}.pth.tar".format(filename)
        if os.path.isfile(filename):
            print("==> Loading from checkpoint '{}'".format(filename))
            try:
                checkpoint = torch.load(filename)
            except:
                checkpoint = pkl.load(open(filename, "rb"))
            epoch = checkpoint["epoch"]
            it = checkpoint.get("it", 0.0)
            best_prec = checkpoint["best_prec"]
            if model is not None and checkpoint["model_state"] is not None:
                model.load_state_dict(checkpoint["model_state"])
            if optimizer is not None and checkpoint["optimizer_state"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("==> Done")
            return it, epoch, best_prec
        else:
            print("==> Checkpoint '{}' not found".format(filename))
            return None

    def define_network(self, weights_path):
        # define model object on GPU
        model = PVN3D(
            num_classes=self.config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
            num_points=self.config.n_sample_points
        ).cuda()
        # convert batch norm into synchornized batch norm
        model = convert_model(model)
        # model to GPU
        model.cuda()
        # load weights
        checkpoint_status = self.load_checkpoint(model, None, filename=weights_path[:-8])
        # convert model to distributed mode
        model = nn.DataParallel(model)
        return model

    def get_normal(self, cld):
        # get normals from point cloud
        cloud = pcl.PointCloud()
        cld = cld.astype(np.float32)
        cloud.from_array(cld)
        ne = cloud.make_NormalEstimation()
        kdtree = cloud.make_kdtree()
        ne.set_SearchMethod(kdtree)
        ne.set_KSearch(50)
        n = ne.compute()
        n = n.to_array()
        return n

    def preprocess_rgbd(self, image, depth, cam_scale=25.0):
        # preprocess RGBD data to be passed to network
        # get camera intrinsics
        K = self.config.intrinsic_matrix['ycb_K1']
        
        # fill missing points in depth map
        dpt = self.bs_utils.fill_missing(depth, cam_scale, 1)

        rgb = np.transpose(image, (2, 0, 1))
        # convert depth map to point cloud
        cld, choose = self.bs_utils.dpt_2_cld(dpt, cam_scale, K)
        normal = self.get_normal(cld)[:, :3]
        normal[np.isnan(normal)] = 0.0

        # construct complete RGB point cloud
        rgb_lst = []
        for ic in range(rgb.shape[0]):
            rgb_lst.append(
                rgb[ic].flatten()[choose].astype(np.float32)
            )
        rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

        choose = np.array([choose])
        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        if len(choose_2) < 400:
            return None
        if len(choose_2) > self.config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.config.n_sample_points-len(choose_2)), 'wrap')

        cld_rgb_nrm = np.concatenate((cld, rgb_pt, normal), axis=1)
        cld = cld[choose_2, :]
        cld_rgb_nrm = cld_rgb_nrm[choose_2, :]
        choose = choose[:, choose_2]

        # define classes indices to be considered
        cls_id_lst = np.array(range(1, 22))

        # convert processed data into torch tensors
        rgb = torch.from_numpy(rgb.astype(np.float32))
        cld = torch.from_numpy(cld.astype(np.float32))
        cld_rgb_nrm = torch.from_numpy(cld_rgb_nrm.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        cls_id_lst = torch.LongTensor(cls_id_lst.astype(np.int32))

        # reshape and copy to GPU
        self.rgb = rgb.reshape((1, rgb.shape[0], rgb.shape[1], rgb.shape[2])).cuda()
        self.cld = cld.reshape((1, cld.shape[0], cld.shape[1])).cuda()
        self.cld_rgb_nrm = cld_rgb_nrm.reshape((1, cld_rgb_nrm.shape[0], cld_rgb_nrm.shape[1])).cuda()
        self.choose = choose.reshape((1, choose.shape[0], choose.shape[1])).cuda()
        self.cls_id_lst = cls_id_lst.reshape((1, cls_id_lst.shape[0]))

    def get_poses(self, save_results=True):
        # perform inference and return objects' poses
        # model to eval mode
        self.model.eval()
        # perform inference on defined model
        with torch.set_grad_enabled(False):
            # network forward pass
            pred_kp_of, pred_rgbd_seg, pred_ctr_of = self.model(
                self.cld_rgb_nrm, self.rgb, self.choose
            )
            _, classes_rgbd = torch.max(pred_rgbd_seg, -1)
            # calculate poses by voting, clustering and linear fitting
            pred_cls_ids, pred_pose_lst = cal_frame_poses(
                self.cld[0], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                self.config.n_objects, True
            )
            # visualize predicted poses
            if save_results:
                np_rgb = self.rgb.cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
                np_rgb = np_rgb[:, :, ::-1].copy()
                ori_rgb = np_rgb.copy()
                # loop over each class id
                for cls_id in self.cls_id_lst[0].cpu().numpy():
                    idx = np.where(pred_cls_ids == cls_id)[0]
                    if len(idx) == 0:
                        continue
                    pose = pred_pose_lst[idx[0]]
                    obj_id = int(cls_id
                    )
                    mesh_pts = self.bs_utils.get_pointxyz(obj_id, ds_type='ycb').copy()
                    mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
                    K = self.config.intrinsic_matrix["ycb_K1"]
                    mesh_p2ds = self.bs_utils.project_p3d(mesh_pts, 1.0, K)
                    color = self.bs_utils.get_label_color(obj_id, n_obj=22, mode=1)
                    np_rgb = self.bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
                # save output visualization
                vis_dir = os.path.join(self.config.log_eval_dir, "pose_vis")
                if not os.path.exists(vis_dir):
                    os.system('mkdir -p {}'.format(vis_dir))
                f_pth = os.path.join(vis_dir, "out.jpg")
                cv2.imwrite(f_pth, np_rgb)
        # return prediction
        return pred_cls_ids, pred_pose_lst

# API test
if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-wp', '--weights_path', type=str, help='path to the pretrained weights file')
    argparser.add_argument('-img', '--image', type=str, help='path to background images for synthetic data')
    argparser.add_argument('-dep', '--depth', type=str, help='number of synthetic training data samples')

    args = argparser.parse_args()

    # read input RGBD image
    image = np.array(Image.open(args.image))
    depth = np.array(Image.open(args.depth))

    # call RGBD Pose API
    pose_estimator = RGBDPoseAPI(args.weights_path)
    pose_estimator.preprocess_rgbd(image, depth)
    pose_estimator.get_poses()
