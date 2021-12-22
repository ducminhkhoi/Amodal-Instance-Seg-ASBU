import json
import argparse
import sys
sys.path.append('.')
import utils
import numpy as np
import pycocotools.mask as maskUtils

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('res', type=str)
    parser.add_argument('ann', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.ann, 'r') as f:
        amodal_data = json.load(f)
    # with open(args.ann, 'r') as f:
    #     annot = json.load(f)

    # print(annot['images'][0].keys())
    # print(annot['annotations'][0].keys())
    print(amodal_data['images'][0].keys())
    print(amodal_data['annotations'][0].keys())
    print(amodal_data.keys())
    # for xs in amodal_data['annotations']:
    #     print('new---------------->', len(xs['regions']), xs['id'])
    #     for x in xs['regions']:
    #         print(x.keys())

    # print(amodal_data['annotations'][0]['regions'][1].keys())

    # categories = {}

    results = []
    for xs in amodal_data['annotations']:
        image_id = xs['image_id']
        for x in xs['regions']:
            data = dict()
            mask = np.array(x['segmentation']).reshape(-1, 2)
            x1, y1, x2, y2 = mask[:, 0].min(), mask[:, 1].min(), mask[:, 0].max(), mask[:, 1].max()
            box = float(x1), float(y1), float(x2-x1), float(y2-y1)

            data['image_id'] = image_id

            # if x['name'] not in categories:
            #     categories[x['name']] = {'supercategory': 'thing', 'id': len(categories), 'name': x['name']}
            
            # data['category_id'] = 1
            
            data['category_id'] = 1 if x['isStuff'] == 1 else 2

            data['inmodal_seg'] = x['visible_mask'] if 'visible_mask' in x else [x['segmentation']]

            data['segmentation'] = [x['segmentation']]
            data['bbox'] = box
            data['area'] = float(box[2] * box[3])
            data['iscrowd'] = 0
            data['id'] = len(results)
            results.append(data)

    # categories = list(categories.values()) 
    # total categories in COCOA: 2140, just use one class as foreground class

    categories = [
        {'supercategory': 'object', 'id': 1, 'name': 'stuff'}, 
        {'supercategory': 'object', 'id': 2, 'name': 'thing'}
    ]

    output_json_dict = dict()
    output_json_dict['images'] = amodal_data['images']
    output_json_dict['categories'] = categories
    output_json_dict['annotations'] = results

    with open(args.output, 'w') as f:
        json.dump(output_json_dict, f)

if __name__ == '__main__':
    main()
