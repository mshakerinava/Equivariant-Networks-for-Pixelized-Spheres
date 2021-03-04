import os
import glob
import gzip
import pickle
import netCDF4
import argparse
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
y = s[:, 0] / np.pi * 768
x = np.where(s[:, 1] < 0, s[:, 1] + 2 * np.pi, s[:, 1]) / (2 * np.pi) * 1152

output_path = os.path.join(args.output_dir, 'climatenet_poly/%s_%d' % (args.poly, args.width))
os.makedirs(output_path, exist_ok=True)
with gzip.open(os.path.join(output_path, 'labels.gz'), 'wb') as f:
    pickle.dump(['<UNK>', 'BG', 'TC', 'AR'], f)

NUM_FACES = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[args.poly]

paths_all = sorted(glob.glob(os.path.join(args.input_dir, 'climatenet/climatenet_data/*.nc')))
paths_split = {
    'train': paths_all[:-18],
    'val': paths_all[-18:-9],
    'test': paths_all[-9:]
}
for key, val in paths_split.items():
    X_cube_list = []
    Y_cube_list = []
    print('processing %s set' % key, flush=True)
    for path in tqdm(val):
        with netCDF4.Dataset(path, 'r') as f:
            data = f.groups['data']
            in_channels = ['TMQ', 'U850', 'V850', 'PRECT']
            X = np.stack([np.array(data.variables[x]) for x in in_channels], axis=2)[None, :, :, :]
            X_cube = bilinear_interpolate(X.astype(np.float64), x, y).astype(np.float32).reshape(1, 6, args.width, args.width, len(in_channels)).transpose(0, 1, 4, 2, 3)
            labels = f.groups['labels']
            ar = np.stack([np.array(labels.variables[x]) for x in list(labels.variables.keys()) if 'ar' in x], axis=2)
            tc = np.stack([np.array(labels.variables[x]) for x in list(labels.variables.keys()) if 'tc' in x], axis=2)
            assert ar.shape == tc.shape
            num_labels = ar.shape[2]
            Y = (1 * tc + 2 * ar).astype(np.uint8)[None, :, :, :]
            Y = np.where(Y == 3, 1, Y)
            Y += 1
            Y_cube = nn_interpolate(Y, x, y).reshape(1, 6, args.width, args.width, num_labels)
        for i in range(num_labels):
            X_cube_list.append(X_cube)
            Y_cube_list.append(Y_cube[:, :, :, :, i])
    print('%s: %d labels' % (key, len(X_cube_list)))
    X_cube_all = np.concatenate(X_cube_list, axis=0)
    Y_cube_all = np.concatenate(Y_cube_list, axis=0)

    with gzip.open(os.path.join(output_path, '%s_data.gz' % key), 'wb') as f:
        pickle.dump(X_cube_all, f)
    with gzip.open(os.path.join(output_path, '%s_labels.gz' % key), 'wb') as f:
        pickle.dump(Y_cube_all, f)
