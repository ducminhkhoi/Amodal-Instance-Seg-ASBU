import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss
from torch.nn import functional as F
import pdb
import math

from . import losses

import matplotlib.pyplot as plt

class PartialCompletionMask(SingleStageModel):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(PartialCompletionMask, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params.get('use_rgb', False)
        self.use_std = params.get('use_std', False)
        self.use_cnp = params.get('use_cnp', False)
        self.loss_mode = params.get('loss_mode', 'gaussian')
        self.return_feat = params.get('return_feat', False)

        outmask_weight = 1.
        if self.use_cnp:
            outmask_weight = 0.

        # loss
        loss_name = params.get('loss_mask', 'MaskWeightedCrossEntropyLoss')

        self.criterion = getattr(losses, loss_name)(
                inmask_weight=params['inmask_weight'], outmask_weight=1.)

        # if self.use_std:
        #     self.criterion = MaskWeightedNLL(
        #         inmask_weight=params['inmask_weight'], outmask_weight=1.)
        # else:
        #     self.criterion = MaskWeightedCrossEntropyLoss(
        #         inmask_weight=params['inmask_weight'], outmask_weight=1.)

    def set_input(self, rgb=None, mask=None, eraser=None, target=None):
        self.eraser_boundary = eraser[:, :1].cuda()
        self.eraser = eraser[:, 1:2].cuda()
        # self.modal_boundary = eraser[:, 2:3].cuda()
        self.target = target.cuda()
        self.rgb = rgb.cuda()
        if self.use_cnp:
            temp = torch.zeros_like(mask)
            temp[mask==1] = 1
            temp[mask==0] = -1
            temp[self.eraser==1] = 0
            self.mask = temp.cuda()
        else:
            self.mask = mask.cuda()

    def evaluate(self, image, inmodal, category, bboxes, amodal, gt_order_matrix, input_size):
        order_method = self.params.get('order_method', 'ours')
        # order
        if order_method == 'ours':
            order_matrix = infer.infer_order2(
                self, image, inmodal, category, bboxes,
                use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_order'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_order', 0),
                input_size=input_size,
                min_input_size=16,
                interp=self.params['inference']['order_interp'])
        elif order_method == 'hull':
            order_matrix = infer.infer_order_hull(inmodal)
        elif order_method == 'area':
            order_matrix = infer.infer_order_area(inmodal, above=self.params['above'])
        elif order_method == 'yaxis':
            order_matrix = infer.infer_order_yaxis(inmodal)
        else:
            raise Exception("No such method: {}".format(order_method))

        gt_order_matrix = infer.infer_gt_order(inmodal, amodal)
        allpair_true, allpair, occpair_true, occpair, show_err = infer.eval_order(
            order_matrix, gt_order_matrix)

        # amodal
        amodal_method = self.params.get('amodal_method', 'ours')
        if amodal_method == 'ours':
            amodal_patches_pred = infer.infer_amodal(
                self, image, inmodal, category, bboxes,
                order_matrix, use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_amodal'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_amodal', 0),
                input_size=input_size,
                min_input_size=16, interp=self.params['inference']['amodal_interp'],
                order_grounded=self.params['inference']['order_grounded'])
            amodal_pred = infer.patch_to_fullimage(
                amodal_patches_pred, bboxes,
                image.shape[0], image.shape[1],
                interp=self.params['inference']['amodal_interp'])
        elif amodal_method == 'hull':
            amodal_pred = np.array(infer.infer_amodal_hull(
                inmodal, bboxes, order_matrix,
                order_grounded=self.params['inference']['order_grounded']))
        elif amodal_method == 'raw':
            amodal_pred = inmodal # evaluate raw
        else:
            raise Exception("No such method: {}".format(amodal_method))

        intersection = ((amodal_pred == 1) & (amodal == 1)).sum()
        union = ((amodal_pred == 1) | (amodal == 1)).sum()
        target = (amodal == 1).sum()

        return allpair_true, allpair, occpair_true, occpair, intersection, union, target

    def forward_only(self, ret_loss=True, val=False):
        with torch.no_grad():
            # if self.use_rgb:
            #     output_ = self.model(torch.cat([self.mask, self.eraser_boundary], dim=1), self.rgb, return_feat=self.return_feat)
            # else:
            #     output_ = self.model(torch.cat([self.mask, self.eraser_boundary], dim=1), return_feat=self.return_feat)

            # if self.return_feat:
            #     output = output_[0].clone()
            # else:
            #     output = output_.clone()

            if self.use_rgb:
                output_ = self.model(torch.cat([self.mask, self.eraser_boundary], dim=1), self.rgb)
            else:
                output_ = self.model(torch.cat([self.mask, self.eraser_boundary], dim=1))

            output = output_.clone()

            if output.shape[2] != self.mask.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.mask.shape[2:4],
                    mode="bilinear", align_corners=True)
        if self.use_std:
            if self.loss_mode == 'gaussian': # modified gaussian distribution
                mean = F.sigmoid(output[:, :1])
                std = F.softplus(output[:, 1:]) + 1e-16
                # std = F.sigmoid(output[:, 1:])
                # std = 1/2*torch.log(2*math.pi*math.e*std.pow(2)) # entropy instead
                comp = (mean > 0.5).float() # optimal threshold for this loss

                # old setting
                # mean, std = output[:, :1], output[:, 1:]
                # comp = (output[:, :1] > 0).float()
                # std_min = -F.adaptive_max_pool2d(-std, 1)
                std_max = F.adaptive_max_pool2d(std, 1)
                # std = (std - std_min) / (std_max - std_min + 1e-16)
                std = std / (std_max + 1e-16)
                ## std[self.eraser==0] = 0 # not use
            else:
                print('not supported loss mode', self.loss_mode) 
                exit()
                
        else:
            comp = output.argmax(dim=1, keepdim=True).float()
        # comp[self.eraser == 0] = (self.mask > 0).float()[self.eraser == 0]
        comp[self.mask > 0] = 1.

        vis_combo = (self.mask > 0).float()
        vis_combo[self.eraser == 1] = 0.5

        if self.target.size(1) == 2:
            vis_target = self.target[:, 0].cpu().clone().float()
        else:
            vis_target = self.target.cpu().clone().float()

        if vis_target.max().item() == 255:
            vis_target[vis_target == 255] = 0.5
        vis_target = vis_target.unsqueeze(1)

        if self.use_rgb and val:
            cm_tensors = [self.rgb.flip([1])]
        else:
            cm_tensors = []
        
        ret_tensors = {'common_tensors': cm_tensors,
                       'mask_tensors':  [self.mask, self.eraser_boundary, mean, std, vis_target, comp, vis_combo] \
                                        if self.use_std else \
                                        [self.mask, self.eraser_boundary, vis_target, comp, vis_combo]}
        if ret_loss:
            loss = self.criterion(output_, self.target, self.eraser.squeeze(1)) / self.world_size
            return ret_tensors, {'loss': loss}
        else:
            return ret_tensors

    def step(self):
        if self.use_rgb:
            output = self.model(torch.cat([self.mask, self.eraser_boundary], dim=1), self.rgb)
        else:
            output = self.model(torch.cat([self.mask, self.eraser_boundary], dim=1))
        
        loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
