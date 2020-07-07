import torch.nn as nn
from .utils import *

class Pose2DLayer(nn.Module):
    def __init__(self, options, is_train):
        super(Pose2DLayer, self).__init__()
        self.coord_norm_factor = 10
        self.keypoints = torch.from_numpy(np.load(options['keypointsfile'])).float()
        self.num_keypoints = int(options['num_keypoints'])
        self.keypoints = self.keypoints[:,:self.num_keypoints,:]
        self.training = is_train

    def forward(self, input, target=None, param = None):
        seen = 0
        if param:
            seen = param[0]

        # input : B x As*(1+2*num_vpoints+num_classes)*H*W
        # output : B x nV x (conf + offset_x + offset_y) x H x W
        t0 = time.time()
        nB = input.data.size(0)
        nA = 1
        nV = self.num_keypoints
        nH = input.data.size(2)
        nW = input.data.size(3)

        output = input.view(nB * nA, (3 * nV), nH * nW).transpose(0, 1). \
            contiguous().view((3 * nV), nB * nA * nH * nW)

        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nA, nH, nW, nV))
        x = output[nV:2*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        y = output[2*nV:3*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)

        grid_x = ((torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nW ) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()
        grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()

        predx = x + grid_x
        predy = y + grid_y

        if self.training:
            # training workflow
            predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
            predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor
            conf = conf.view(nB, nH, nW, nV)
            return [predx, predy, conf]
        else:
            # inference workflow
            predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
            predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor

            # copy to CPU
            conf = convert2cpu(conf.view(nB,nH,nW,nV)).detach().numpy()
            px = convert2cpu(predx).detach().numpy()
            py = convert2cpu(predy).detach().numpy()
            keypoints = convert2cpu(self.keypoints).detach().numpy()
            
            out_preds = [px, py, conf, keypoints]
            return out_preds
