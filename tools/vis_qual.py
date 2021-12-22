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
            size = max([np.sqrt(bbox[2] * bbox[3] * self.args.data['enlarge_box']),
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

        thres = 5000
        folder = f'qualitative_results_boxes_{thres}'
        os.makedirs(folder, exist_ok=True)

        count = 0

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
            ori_boxes = bboxes
            bboxes = self.expand_bbox(bboxes)

            # gt order
            gt_order_matrix = infer.infer_gt_order(modal, amodal_gt)

            # infer order
            if self.args.order_method == 'area':
                order_matrix = infer.infer_order_area(
                    modal, above='smaller' if self.args.data['dataset'] == 'COCOA' else 'larger')

            elif self.args.order_method == 'yaxis':
                order_matrix = infer.infer_order_yaxis(modal)

            elif self.args.order_method == 'convex':
                order_matrix = infer.infer_order_convex(modal)

            elif self.args.order_method == 'ours':
                order_matrix = infer.infer_order(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=order_th, dilate_kernel=args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False, args=self.args)

            elif self.args.order_method == 'sup': # supervised
                order_matrix = infer.infer_order(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=order_th, dilate_kernel=args.dilate_kernel,
                    input_size=256, min_input_size=16, interp='nearest', debug_info=False, args=self.args, supervised=True)
            else:
                raise Exception('No such order method: {}'.format(self.args.order_method))

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
                inmodal_patches, eraser_patches, amodal_patches_pred, amodal_patches_gt, std_patches, boundary_patches, image_patches = \
                    infer.infer_amodal_vis(
                        self.model, image, modal, amodal_gt, category, bboxes, order_matrix,
                        use_rgb=self.args.model['use_rgb'], th=amodal_th, dilate_kernel=args.dilate_kernel,
                        input_size=256, min_input_size=16, interp='linear',
                        order_grounded=True, debug_info=False, args=args)

                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp='linear')

            elif self.args.amodal_method == 'sup': # supervised
                amodal_patches_pred = infer.infer_amodal_sup(
                    self.model, image, modal, category, bboxes,
                    use_rgb=self.args.model['use_rgb'], th=amodal_th, input_size=256,
                    min_input_size=16, interp='linear', args=args)
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

            indices1 = (gt_order_matrix[0]==1).nonzero()[0].tolist()
            indices2 = (gt_order_matrix[:, 0]==1).nonzero()[0].tolist()

            indices = indices1 + indices2

            for k, i_object in enumerate(indices):
                modal_i, amodal_i, box_i, cat_i = modal[i_object], amodal_pred[i_object], ori_boxes[i_object], category[i_object]

                inmodal, eraser, amodal, amodal_gt, std, boundary, img = [x[i_object] for x in [inmodal_patches, eraser_patches,
                                                amodal_patches_pred, amodal_patches_gt, std_patches, boundary_patches, image_patches]]

                # print(i, k, modal.shape, len(image_patches), amodal.sum() - inmodal.sum(), modal_i.sum())
                if self.dataset == 'KINS' and cat_i in [1, 2]:
                    thres_2 = 1000
                    thres_1 = 200
                    text = '_person'
                else:
                    thres_2 = thres
                    thres_1 = 1000
                    text = ''

                # if amodal.sum() - inmodal.sum() > 1000 and amodal_i.sum() > thres: # just show the large enough examples
                if amodal.sum() - inmodal.sum() > thres_1 and box_i[2] * box_i[3] > thres_2: # just show the large enough examples

                    inmodal, eraser, amodal, amodal_gt, std, boundary = [x[..., None] * np.array([[[255, 255, 255]]]) 
                                                                for x in [inmodal, eraser, amodal, amodal_gt, std, boundary]]

                    show_image = np.concatenate([img[:, :, ::-1], inmodal, boundary, amodal, std, amodal_gt], 1)

                    print(count, text)

                    cv2.imwrite(f'{folder}/{self.dataset}_{count}{text}.jpg', show_image)
                    count += 1

                    if count >= 100:
                        exit()


if __name__ == "__main__":
    args = parse_args()
    main(args)
