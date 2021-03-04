import os
import glob
import gzip
import pickle
import argparse
import h5py as h5
import numpy as np
from tqdm import tqdm
from utils import *
from poly_sphere import *


parser = argparse.ArgumentParser()
parser.add_argument('--poly', choices=['icosa', 'octa', 'cube', 'tetra'], required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--input-dir', type=str, default='.')
parser.add_argument('--output-dir', type=str, default='.')
args = parser.parse_args()

u = get_sampling_grid(args.poly, args.width)
u = u.reshape(-1, 3)
s = cartesian_to_spherical(u)[:, 1:]

output_path = os.path.join(args.output_dir, 'happi20_poly/%s_%d' % (args.poly, args.width))
os.makedirs(output_path, exist_ok=True)
with gzip.open(os.path.join(output_path, 'labels.gz'), 'wb') as f:
    pickle.dump(['<UNK>', 'BG', 'TC', 'AR'], f)

NUM_FACES = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[args.poly]

for t in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_path, t), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, 'happi20/%s/*.h5' % t)))
    print('processing %s set (%d files)' % (t, len(paths)), flush=True)
    for i, path in enumerate(tqdm(paths)):
        fin = h5.File(path, mode='r')
        img = np.array(fin['climate']['data']).transpose(1, 2, 0)
        labels = np.array(fin['climate']['labels'])
        X = img[None, :, :, :]
        Y = labels[None, :, :].astype(np.uint8) + 1
        X_poly = equirectangular_to_polysphere(X, s, interpolation='bilinear')
        X_poly = X_poly.reshape(NUM_FACES, args.width, args.width, 16).transpose(0, 3, 1, 2)
        Y_poly = equirectangular_to_polysphere(Y, s, interpolation='nearest')
        Y_poly = Y_poly.reshape(NUM_FACES, args.width, args.width)

        output_path_X = os.path.join(output_path, t, '%05d.x.npy' % i)
        output_path_Y = os.path.join(output_path, t, '%05d.y.npy' % i)
        with open(output_path_X, 'wb') as f:
            np.save(f, X_poly)
        with open(output_path_Y, 'wb') as f:
            np.save(f, Y_poly)
