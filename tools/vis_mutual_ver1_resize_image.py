import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
sys.path.append('.')
from datasets import reader
import models
import inference as infer
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--load-model', required=True, type=str)
    parser.add_argument('--order-method', required=True, type=str)
    parser.add_argument('--amodal-method', required=True, type=str)
    parser.add_argument('--order-th', default=0.1, type=float)
    parser.add_argument('--amodal-th', default=0.2, type=float)
    parser.add_argument('--annotation', required=True, type=str)
    parser.add_argument('--image-root', required=True, type=str)
    parser.add_argument('--test-num', default=-1, type=int)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--dilate_kernel', default=0, type=int)
    parser.add_argument('--scale', default=3, type=int)
    parser.add_argument('--resize_image', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()

class Tester(object):
    def __init__(self, args):
        self.args = args
        self.prepare_data()

    def prepare_data(self):
        config = self.args.data
        dataset = config['dataset']
        self.data_root = self.args.image_root
        if dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(self.args.annotation)
        else:
            if dataset == 'KINSNew':
                self.data_reader = reader.KINSNewDataset(
                    dataset, self.args.annotation)
            else:
                self.data_reader = reader.KINSLVISDataset(
                    dataset, self.args.annotation)
        self.data_length = self.data_reader.get_image_length()
        self.dataset = dataset
        if self.args.test_num != -1:
            self.data_length = self.args.test_num

    def prepare_model(self):
        self.model = models.__dict__[self.args.model['algo']](self.args.model, dist_model=False)
        self.model.load_state(self.args.load_model)
        self.model.switch_to('eval')

    def expand_bbox(self, bboxes):
        new_bboxes = []
        for bbox in bboxes:
            centerx = bbox[0] + bbox[2] / 2.
            centery = bbox[1] + bbox[3] / 2.
            size = max([np.sqrt(bbox[2] * bbox[3] * self.args.scale),
                        bbox[2] * 1.1, bbox[3] * 1.1])
            new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
            new_bboxes.append(new_bbox)
        return np.array(new_bboxes)

    def run(self):
        self.prepare_model()
        self.infer()

    def infer(self):
        order_th = self.args.order_th
        amodal_th = self.args.amodal_th

        self.args.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.args.data['data_mean'], self.args.data['data_std'])
            ])

        segm_json_results = []
        self.count = 0
        
        allpair_true_rec = utils.AverageMeter()
        allpair_rec = utils.AverageMeter()
        occpair_true_rec = utils.AverageMeter()
        occpair_rec = utils.AverageMeter()
        intersection_rec = utils.AverageMeter()
        union_rec = utils.AverageMeter()
        target_rec = utils.AverageMeter()

        # for i in tqdm(range(self.data_length), total=self.data_length):
        for i in range(self.data_length):
            modal, category, bboxes, amodal_gt, image_fn = self.data_reader.get_image_instances(
                i, with_gt=True)

            # data
            image = Image.open(os.path.join(self.data_root, image_fn)).convert('RGB')
            if image.size[0] != modal.shape[2] or image.size[1] != modal.shape[1]:
                image = image.resize((modal.shape[2], modal.shape[1]))

            image = np.array(image)
            h, w = image.shape[:2]
            bboxes = self.expand_bbox(bboxes)

            category = np.ones_like(category)

            # Handle the case when the object is too big compared to the image size
            if self.args.resize_image:
                ori_image = image
                ori_h = h
                ori_w = w
                ori_modal = modal

                new_im_size = 5000
                start = 1500
                new_image = np.zeros((new_im_size, new_im_size, 3)).astype('uint8')
                new_image[start:start+h, start:start+w] = image
                image = new_image

                new_modal = np.zeros((len(modal), new_im_size, new_im_size))
                new_modal[:, start:start+h, start:start+w] = modal
                modal = new_modal
                
                bboxes[:, :2] += start
                h = new_im_size
                w = new_im_size            

                print(bboxes)

            order_matrix = np.array([[0, 1], [1, 0]])

            # infer amodal
            if self.args.amodal_method == 'raw':
                amodal_pred = modal.copy()

            elif self.args.amodal_method == 'ours_nog':
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded=False, debug_info=False, args=args)
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'ours_parents':
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded='parents', debug_info=False, args=args)
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'ours':
                amodal_patches_pred = infer.infer_amodal(
                    self.model, image, modal, category, bboxes, order_matrix,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='linear',
                    order_grounded=True, debug_info=False, args=args)
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'sup': # supervised
                amodal_patches_pred = infer.infer_amodal_sup(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, input_size=256,
                    min_input_size=16, interp='linear')
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'convex':
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, None, order_grounded=False))

            elif self.args.amodal_method == 'convexr':
                order_matrix = infer.infer_order_hull(modal)
                amodal_pred = np.array(infer.infer_amodal_hull(
                    modal, bboxes, order_matrix, order_grounded=True))

            else:
                raise Exception("No such method: {}".format(self.args.method))

            ## visualize the mutual occlusion
            if self.args.resize_image:
                image = ori_image
                amodal_pred = amodal_pred[:, start:start+ori_h, start:start+ori_w]
                modal = ori_modal

            new_image = image.copy()
            modal_1 = modal[0][..., None]
            modal_2 = modal[1][..., None]
            new_image = new_image * (1 - modal_1) + np.array([[[0, 0, 255]]]) * modal_1
            new_image = new_image * (1 - modal_2) + np.array([[[255, 0, 0]]]) * modal_2
            cv2.imwrite(f'mutual/{image_fn}_annot.jpg', new_image)

            new_image = np.array([[[0, 0, 255]]]) * amodal_pred[0][..., None]
            cv2.imwrite(f'mutual/{image_fn}_amodal_1.jpg', new_image)

            new_image = np.array([[[255, 0, 0]]]) * amodal_pred[1][..., None]
            cv2.imwrite(f'mutual/{image_fn}_amodal_2.jpg', new_image)

            plt.close('all')


if __name__ == "__main__":
    args = parse_args()
    main(args)
