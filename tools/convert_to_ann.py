import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('res', type=str)
    parser.add_argument('ann', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.res, 'r') as f:
        data = json.load(f)
    with open(args.ann, 'r') as f:
        annot = json.load(f)

    print(data[0].keys())
    exit()

    output_json_dict = dict()
    output_json_dict['images'] = annot['images']
    # output_json_dict['categories'] = annot['categories'] # for KINS
    # output_json_dict['categories'] = [{'supercategory': 'thing', 'id': 1, 'name': 'thing'}] # for COCOA thing class only 
    output_json_dict['categories'] = [{'supercategory': 'object', 'id': 1, 'name': 'stuff'}, {'supercategory': 'object', 'id': 2, 'name': 'thing'}] # for COCOA stuff and thing class only 
    output_json_dict['annotations'] = data

    with open(args.output, 'w') as f:
        json.dump(output_json_dict, f)

if __name__ == '__main__':
    main()
