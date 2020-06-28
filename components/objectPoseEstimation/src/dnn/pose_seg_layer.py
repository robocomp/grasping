import torch.nn as nn
import torch.nn.functional as F
from utils import *

class PoseSegLayer(nn.Module):
    def __init__(self, options, is_train):
        super(PoseSegLayer, self).__init__()
        self.num_classes = int(options['classes'])
        self.training = is_train

    def forward(self, input):
        # input : B x As*(1+2*num_vpoints+num_classes)*H*W
        # output : nB x nC x H x W
        nB = input.data.size(0)
        nA = 1
        nC = self.num_classes
        nH = input.data.size(2)
        nW = input.data.size(3)

        # update object_scale according to nA and nH and nW
        # self.object_scale = nA * nH * nW * 0.01

        output = input.view(nB * nA, (nC), nH * nW).transpose(0, 1). \
            contiguous().view((nC), nB * nA * nH * nW)

        cls = output[0:nC].transpose(0, 1)
        t1 = time.time()

        if self.training:
            # training workflow
            output = output.transpose(0, 1)
            return output
        else:
            # inference workflow
            cls_confs, cls_ids = torch.max(F.softmax(cls, 1), 1)
            cls_confs = cls_confs.view(nB, nH, nW)
            cls_ids = cls_ids.view(nB, nH, nW)

            # copy to CPU
            cls_confs = convert2cpu(cls_confs).detach().numpy()
            cls_ids = convert2cpu_long(cls_ids).detach().numpy()

            out_preds = [cls_confs, cls_ids]
            return out_preds
