import multiprocessing as mp
import tqdm
import numpy as np
import json
import sys
import pycocotools.mask as maskUtils

def read_annot(ann, h, w):
    segm = ann['inmodal_seg']
    if isinstance(segm, list):
        modal = maskUtils.decode(maskUtils.frPyObjects(segm, h, w))
    else:
        modal = maskUtils.decode(segm)

    amodal = maskUtils.decode(maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
    return modal, amodal

def task(ann, data, size_dict):
    w, h = size_dict[ann['image_id']]
    amp = maskUtils.decode(data['segmentation']).astype(np.bool)
    m, amg = read_annot(ann, h, w)
    return [((amp == 1) & (amg == 1)).sum(),
            ((amp == 1) | (amg == 1)).sum(),
            m.sum(), amg.sum()]

def helper(args):
    return task(*args)

def compute(data, annot_data, size_dict):
    num = len(data)
    pool = mp.Pool(16)
    args = zip(annot_data, data, [size_dict] * num)
    ret = list(tqdm.tqdm(pool.imap(helper, args), total=num))
    return np.array(ret) # Nx4

if __name__ == "__main__":
    method = 'std_no_rgb_mumford_shah'
    test_set = 'val'
    res_data = json.load(open(f'experiments/COCOA/pcnet_m_{method}/amodal_results/amodalcomp_{test_set}_ours.json', 'r'))
    annot_data = json.load(open(f'data/COCOA/annotations/amodal_{test_set}2014_new.json', 'r'))

    size_dict = dict([(a['id'], (a['width'], a['height'])) for a in annot_data['images']])

    ret = compute(res_data, annot_data['annotations'], size_dict)

    np.save(f"experiments/COCOA/stats/stat_{method}_{test_set}.npy", ret)