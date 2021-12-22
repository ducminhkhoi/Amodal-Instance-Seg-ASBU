import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader
import matplotlib.pyplot as plt
import kornia
from torch.nn import functional as F

class PartialCompDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        else:
            if self.dataset == 'KINSNew':
                self.data_reader = reader.KINSNewDataset(
                    self.dataset, config['{}_annot_file'.format(phase)])
            else:
                self.data_reader = reader.KINSLVISDataset(
                    self.dataset, config['{}_annot_file'.format(phase)])

        self.use_rgb = config['load_rgb']
        if self.use_rgb:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['data_mean'], config['data_std'])
            ])
        self.eraser_setter = utils.EraserSetterRGB(config['eraser_setter'])
        self.sz = config['input_size']
        self.eraser_front_prob = config['eraser_front_prob']
        self.phase = phase
        self.use_default = config['use_default']
        self.use_matting = config.get('use_matting', False)
        self.border_width = config.get('border_width', 5)
        self.occluded_only = config.get('occluded_only', False)
        self.boundary_label = config.get('boundary_label', False)

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)
        self.edge_detection = kornia.filters.Sobel()

    def __len__(self):
        return self.data_reader.get_instance_length()

    def _init_memcached(self):
        if not self.initialized:
            assert self.memcached_client is not None, "Please specify the path of your memcached_client"
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
            except:
                print('Read image failed ({})'.format(fn))
                raise Exception("Exit")
            else:
                return img
        else:
            try:
                return Image.open(fn).convert('RGB')
            except: 
                return Image.open(fn.replace('val2017', 'train2017')).convert('RGB')

    def _get_inst(self, idx, load_rgb=False, randshift=False):
        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)
        centerx = bbox[0] + bbox[2] / 2.
        centery = bbox[1] + bbox[3] / 2.
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])
        if size < 5 or np.all(modal == 0):
            return self._get_inst(
                np.random.choice(len(self)), load_rgb=load_rgb, randshift=randshift)

        # shift & scale aug
        if self.phase  == 'train':
            if randshift:
                centerx += np.random.uniform(*self.config['base_aug']['shift']) * size
                centery += np.random.uniform(*self.config['base_aug']['shift']) * size
            size /= np.random.uniform(*self.config['base_aug']['scale'])

        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal = modal[:, ::-1]
        else:
            flip = False

        if load_rgb:
            rgb = np.array(self._load_image(os.path.join(
                self.config['{}_image_root'.format(self.phase)], imgfn))) # uint8
            rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0,0,0)),
                (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)
            if flip:
                rgb = rgb[:, ::-1, :]

        if load_rgb:
            return modal, category, rgb
        else:
            return modal, category, None

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()
        randidx = np.random.choice(len(self))
        modal, category, rgb = self._get_inst(
            idx, load_rgb=True, randshift=True) # modal, uint8 {0, 1} # consider not to use shift in our approach
        if not self.config.get('use_category', True):
            category = 1

        eraser, _, eraser_rgb = self._get_inst(randidx, load_rgb=True, randshift=False)

        eraser, eraser_rgb = self.eraser_setter(modal, eraser, eraser_rgb) # uint8 {0, 1}

        border_width = self.border_width

        # erase
        erased_modal = modal.copy()
        eraser_above = np.random.rand() < self.eraser_front_prob

        if eraser_above:
            eraser_mask = eraser
            erased_modal[eraser == 1] = 0 # eraser above modal
            
            if self.occluded_only or self.boundary_label:
                occluded = (eraser == 1) & (modal == 1)

            if self.boundary_label:
                occluded_extend = F.max_pool2d(torch.from_numpy(occluded[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
                complement_extend = F.max_pool2d(torch.from_numpy((1-modal)[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
                gt_boundary = ((occluded_extend == 1) & (complement_extend == 1))[0, 0].float()

        else:
            eraser_mask = (eraser == 1) & (modal == 0) # B \ A
            eraser[modal == 1] = 0 # eraser below modal

            if self.occluded_only:
                occluded = np.zeros_like(erased_modal)

            if self.boundary_label:
                gt_boundary = torch.zeros_like(torch.from_numpy(erased_modal)).float()

        eraser_mask = eraser_mask.astype('float') # just used for matting RGB image

        eraser_tensor = torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0) # 1HW

        if self.use_default:
            keep_bounary = eraser_tensor  
        else:
            eraser_extend = F.max_pool2d(torch.from_numpy(eraser[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
            modal_extend = F.max_pool2d(torch.from_numpy(erased_modal[None, None, ...]).float(), border_width, stride=1, padding=border_width//2)
            keep_bounary = ((eraser_extend == 1) & (modal_extend == 1))[0].float()

            # image matting boundary
            eraser_mask[keep_bounary[0].numpy()==1] = 0.5

            eraser_mask[eraser_mask==1] = 0.8 # matting almost for other pixels, change from 0.8

        eraser_tensor = torch.cat([keep_bounary, eraser_tensor]) # HW

        erased_modal = erased_modal.astype(np.float32) * category
        # erase rgb
        if rgb is not None and self.use_rgb:
            if self.use_matting:
                eraser_mask = eraser_mask[..., None]
                # rgb = (rgb * 1 - eraser_mask) + eraser_rgb * (eraser_mask)
                rgb = rgb * (1 - eraser_mask) + eraser_rgb * eraser_mask
            else:
                rgb = rgb * (1 - eraser_tensor[1, ..., None].numpy())

            rgb = self.img_transform(rgb).float() # CHW
        else:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32) # 3HW

        erased_modal_tensor = torch.from_numpy(
            erased_modal.astype(np.float32)).unsqueeze(0) # 1HW
        
        if self.occluded_only:
            target = torch.from_numpy(occluded.astype(np.int))
        else:
            target = torch.from_numpy(modal.astype(np.int)) # HW

        if self.boundary_label:
            target = torch.stack([target, gt_boundary.long()])

        return rgb, erased_modal_tensor, eraser_tensor, target
