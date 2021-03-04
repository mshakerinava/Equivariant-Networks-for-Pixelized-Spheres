import os
import json
import glob
import gzip
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import *
from poly_sphere import *


parser = argparse.ArgumentParser()
parser.add_argument('--poly', choices=['icosa', 'octa', 'cube', 'tetra'], required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--input-dir', type=str, default='.')
parser.add_argument('--output-dir', type=str, default='.')
args = parser.parse_args()


def load_labels(label_file):
    """ Convenience function for loading JSON labels """
    with open(label_file) as f:
        return json.load(f)


def parse_label(label):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split("_")
    res['instance_class'] = clazz
    res['instance_num'] = int(instance_num)
    res['room_type'] = room_type
    res['room_num'] = int(room_num)
    res['area_num'] = int(area_num)
    return res


semantic_labels = load_labels(os.path.join(args.input_dir, 'stanford_2d3ds/assets/semantic_labels.json'))

n = len(semantic_labels)
col_to_label = np.zeros(n, dtype=int)
label_to_idx = {}

cnt = 0
for i in range(n):
    x = parse_label(semantic_labels[i])
    label = x['instance_class']
    if label not in label_to_idx:
        label_to_idx[label] = cnt
        cnt += 1
    idx = label_to_idx[label]
    col_to_label[i] = idx


num_labels = cnt
idx_to_label = [None] * num_labels
for k, v in label_to_idx.items():
    idx_to_label[v] = k

assert idx_to_label == ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']

u = get_sampling_grid(args.poly, args.width)
u = u.reshape(-1, 3)
s = cartesian_to_spherical(u)[:, 1:]

output_path = os.path.join(args.output_dir, '2d3ds_poly/%s_%d' % (args.poly, args.width))
os.makedirs(output_path, exist_ok=True)
with gzip.open(os.path.join(output_path, 'labels.gz'), 'wb') as f:
    pickle.dump(idx_to_label, f)

NUM_FACES = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[args.poly]

for area in range(1, 7):
    print('processing area %d/6' % area, flush=True)
    paths_rgb = glob.glob(os.path.join(args.input_dir, 'stanford_2d3ds/area_%d*/pano/rgb/*.png' % area))
    X_poly_list = []
    Y_poly_list = []
    for path_rgb in tqdm(paths_rgb):
        path_d = path_rgb.replace('rgb', 'depth')
        path_s = path_rgb.replace('rgb', 'semantic')
        img_rgb = np.array(Image.open(path_rgb))
        img_d = np.array(Image.open(path_d))
        img_s = np.array(Image.open(path_s), dtype=int)
        X = np.concatenate((img_rgb[:, :, :3], img_d[:, :, None]), axis=2)[None, :, :, :]
        Y = img_s[:, :, 0] * 256 * 256 + img_s[:, :, 1] * 256 + img_s[:, :, 2]
        Y = np.where(Y == 855309, 0, Y)
        Y = col_to_label[Y]
        assert (Y < 14).all()
        Y = Y.astype(np.uint8)[None, :, :, None]
        X_poly = equirectangular_to_polysphere(X, s, interpolation='bilinear')
        X_poly = X_poly.reshape(1, NUM_FACES, args.width, args.width, 4).transpose(0, 1, 4, 2, 3)
        Y_poly = equirectangular_to_polysphere(Y, s, interpolation='nearest')
        Y_poly = Y_poly.reshape(1, NUM_FACES, args.width, args.width)
        if args.poly != 'cube':
            X_poly = X_poly * triangle_mask(args.width)[None, None, None, :, :]
            Y_poly = Y_poly * triangle_mask(args.width)[None, None, :, :]
        X_poly_list.append(X_poly)
        Y_poly_list.append(Y_poly)
    X_poly_all = np.concatenate(X_poly_list, axis=0)
    Y_poly_all = np.concatenate(Y_poly_list, axis=0)
    gzip.open(os.path.join(output_path, 'labels.gz'), 'wb')
    with gzip.open(os.path.join(output_path, 'area_%d_rgbd_images.gz' % area), 'wb') as f:
        pickle.dump(X_poly_all, f)
    with gzip.open(os.path.join(output_path, 'area_%d_semantic_images.gz' % area), 'wb') as f:
        pickle.dump(Y_poly_all, f)
