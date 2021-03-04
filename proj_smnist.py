import os
import gzip
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from utils import *
from poly_sphere import *


parser = argparse.ArgumentParser()
parser.add_argument('--poly', choices=['icosa', 'octa', 'cube', 'tetra'], required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--chunk-size', type=int, default=500)
args = parser.parse_args()


NUM_FACES = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[args.poly]

print('Reading `%s`...' % args.input)
with gzip.open(args.input, 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['train']['images']
Y_train = dataset['train']['labels']
X_test = dataset['test']['images']
Y_test = dataset['test']['labels']

# add south pole
south_pole = np.zeros_like(X_train[:, -1:, :]) + np.mean(X_train[:, -1:, :], axis=2, keepdims=True).astype(X_train.dtype)
X_train = np.concatenate([X_train, south_pole], axis=1)
south_pole = np.zeros_like(X_test[:, -1:, :]) + np.mean(X_test[:, -1:, :], axis=2, keepdims=True).astype(X_test.dtype)
X_test = np.concatenate([X_test, south_pole], axis=1)

u = get_sampling_grid(args.poly, args.width)
u = u.reshape(-1, 3)
s = cartesian_to_spherical(u)[:, 1:]

CHUNK = args.chunk_size

print('Interpolating Training Set...')
m = 60000
X_proj_train_list = []
for i in tqdm(range(m // CHUNK)):
    t = equirectangular_to_polysphere(X_train[i * CHUNK: (i + 1) * CHUNK, :, :, None], s, interpolation='bilinear')
    t = t.reshape(CHUNK, NUM_FACES, 1, args.width, args.width)
    if args.poly != 'cube':
        t = t * triangle_mask(args.width)[None, None, None, :, :]
    X_proj_train_list.append(t)
X_proj_train = np.concatenate(X_proj_train_list, axis=0)
assert X_proj_train.shape == (m, NUM_FACES, 1, args.width, args.width)

print('Interpolating Test Set...')
m = 10000
X_proj_test_list = []
for i in tqdm(range(m // CHUNK)):
    t = equirectangular_to_polysphere(X_test[i * CHUNK: (i + 1) * CHUNK, :, :, None], s, interpolation='bilinear')
    t = t.reshape(CHUNK, NUM_FACES, 1, args.width, args.width)
    if args.poly != 'cube':
        t = t * triangle_mask(args.width)[None, None, None, :, :]
    X_proj_test_list.append(t)
X_proj_test = np.concatenate(X_proj_test_list, axis=0)
assert X_proj_test.shape == (m, NUM_FACES, 1, args.width, args.width)

os.makedirs(args.output, exist_ok=True)
print('Writing in `%s`...' % args.output)

with gzip.open(os.path.join(args.output, 'train-images-idx3-ubyte.gz'), 'wb') as f:
    pickle.dump(X_proj_train, f)

with gzip.open(os.path.join(args.output, 't10k-images-idx3-ubyte.gz'), 'wb') as f:
    pickle.dump(X_proj_test, f)

with gzip.open(os.path.join(args.output, 'train-labels-idx1-ubyte.gz'), 'wb') as f:
    pickle.dump(Y_train, f)

with gzip.open(os.path.join(args.output, 't10k-labels-idx1-ubyte.gz'), 'wb') as f:
    pickle.dump(Y_test, f)
