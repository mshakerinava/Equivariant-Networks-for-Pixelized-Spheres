import os
import io
import sys
import json
import gzip
import zlib
import pickle
import random
import base64
import argparse
import subprocess
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import hash_args
from layers import *
from nets import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='.')
parser.add_argument('--checkpoint_path', type=str, required=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--normalize_data', type=str2bool, default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--gpu', type=str2bool, default=True)
parser.add_argument('--use_batch_norm', type=str2bool, default=True)
parser.add_argument('--use_cube_pad', type=str2bool, default=False)
parser.add_argument('--act_fn', type=str, default='ReLU')
parser.add_argument('--pooling', type=str, choices=['max', 'avg'], default='max')
parser.add_argument('--polyhedron', type=str, choices=['icosa', 'cube', 'octa', 'tetra'], required=True)
parser.add_argument('--model_type', type=int, choices=[1, 2], default=2)
parser.add_argument('--frac_hier', type=float, default=0.25)
parser.add_argument('--oriented', type=str2bool, default=False)
parser.add_argument('--attention', type=str2bool, default=False)
parser.add_argument('--l2_decay', type=float, default=0)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--sched_step_size', type=int, default=20)
parser.add_argument('--sched_gamma', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--num_epochs', type=int, default=50, help='set negative value to use an adaptive number of epochs')
parser.add_argument('--only_profile', type=str2bool, default=False)
parser.add_argument('--use_dataparallel', type=str2bool, default=False)
parser.add_argument('--use_weighted_loss', type=str2bool, default=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--num_channels', type=int, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--fold', type=int, choices=[1, 2, 3], required=False, help='only relevant when dataset is stanford 2d3ds')
args = parser.parse_args()

nn.Swish = Swish

commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
code_hash = subprocess.check_output('cat *.py autoequiv/*.py | md5sum', shell=True).decode('utf-8').split()[0].strip()
args_hash = hash_args(vars(args), no_hash=['seed', 'checkpoint_path', 'base_path'])

FINISH_TEXT = '** finished successfully **'
TAG = 'args-%s__seed-%02d__code-%s' % (args_hash, args.seed, code_hash[:8])

try:
    old_print
except NameError:
    os.makedirs(os.path.join(args.base_path, 'logs'), exist_ok=True)
    LOG_PATH = os.path.join(args.base_path, 'logs', TAG + '.txt')

    abort = False
    try:
        if subprocess.check_output(['tail', '-n', '1', LOG_PATH]).decode('utf-8').strip() == FINISH_TEXT:
            print('ABORT: experiment has already been performed')
            abort = True
    except:
        pass

    if abort:
        sys.exit(-1)

    LOG_STR = io.StringIO()
    LOG_FILE = open(LOG_PATH, 'w')

    old_print = print

    def print(*args, **kwargs):
        kwargs['flush'] = True
        old_print(*args, **kwargs)
        kwargs['file'] = LOG_STR
        old_print(*args, **kwargs)
        kwargs['file'] = LOG_FILE
        old_print(*args, **kwargs)

print('writing log to `%s`' % LOG_PATH)
print('commit %s' % commit_hash)
print('code hash %s' % code_hash)

print('───────────── machine info ─────────────')
print(subprocess.check_output(['uname', '-a']).decode('utf-8').strip())
print(subprocess.check_output(['lscpu']).decode('utf-8').strip())
print(subprocess.check_output(['nvidia-smi']).decode('utf-8').strip())
print('────────────────────────────────────────')

print(subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip())
print('args = %s' % json.dumps(vars(args), sort_keys=True, indent=4))


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ensure reproducibility (?)
set_seed(args.seed)

DATASET_PATH = os.path.join(args.base_path, args.dataset)

if args.task == 'segmentation':
    if '2d3ds' in args.dataset.lower():
        # Stanford 2D3DS: 3-fold cross-validation scheme
        # ┌──────┬───────────────┬─────────┐
        # | Fold |   Training    | Testing |
        # ├──────┼───────────────┼─────────┤
        # | 1    | 1, 2, 3, 4, 6 | 5       |
        # | 2    | 1, 3, 5, 6    | 2, 4    |
        # | 3    | 2, 4, 5       | 1, 3, 6 |
        # └──────┴───────────────┴─────────┘

        if args.fold == 1:
            areas_train = [1, 2, 3, 4, 6]
            areas_val = [5]
        elif args.fold == 2:
            areas_train = [1, 3, 5, 6]
            areas_val = [2, 4]
        elif args.fold == 3:
            areas_train = [2, 4, 5]
            areas_val = [1, 3, 6]
        else:
            assert False, 'ERROR: invalid argument for `--fold`'

        assert len(areas_train) + len(areas_val) == 6
        assert sorted(areas_train + areas_val) == list(range(1, 7))

        X_list = [None] * 7
        Y_list = [None] * 7

        for i in range(1, 7):
            with gzip.open(os.path.join(DATASET_PATH, 'area_%d_rgbd_images.gz' % i), 'rb') as f:
                X_list[i] = pickle.load(f).astype(np.float32) / 255

            with gzip.open(os.path.join(DATASET_PATH, 'area_%d_semantic_images.gz' % i), 'rb') as f:
                Y_list[i] = pickle.load(f)

        X_train = np.concatenate([X_list[i] for i in areas_train], axis=0)
        Y_train = np.concatenate([Y_list[i] for i in areas_train], axis=0)
        X_val = np.concatenate([X_list[i] for i in areas_val], axis=0)
        Y_val = np.concatenate([Y_list[i] for i in areas_val], axis=0)
        X_test = X_val
        Y_test = Y_val

        label_ratio = [0.014504436907968913, 0.017173225930738712, 
                       0.048004778186652164, 0.17384037404789865, 0.028626771620973622, 
                       0.087541966989014, 0.019508096683310605, 0.08321331842901526, 
                       0.17002664771895903, 0.002515611224467519, 0.020731298851232174, 
                       0.2625963729249342, 0.016994731594287146]
        label_weight = 1 / np.log(1.02 + np.array(label_ratio))
        label_weight = label_weight.astype(np.float32)

    elif 'climatenet' in args.dataset.lower():
        with gzip.open(os.path.join(DATASET_PATH, 'train_data.gz'), 'rb') as f:
            X_train = pickle.load(f)

        with gzip.open(os.path.join(DATASET_PATH, 'train_labels.gz'), 'rb') as f:
            Y_train = pickle.load(f)

        with gzip.open(os.path.join(DATASET_PATH, 'val_data.gz'), 'rb') as f:
            X_val = pickle.load(f)

        with gzip.open(os.path.join(DATASET_PATH, 'val_labels.gz'), 'rb') as f:
            Y_val = pickle.load(f)

        with gzip.open(os.path.join(DATASET_PATH, 'test_data.gz'), 'rb') as f:
            X_test = pickle.load(f)

        with gzip.open(os.path.join(DATASET_PATH, 'test_labels.gz'), 'rb') as f:
            Y_test = pickle.load(f)

        label_ratio = [np.sum(Y_train == i) for i in range(1, 4)]
        label_ratio = label_ratio / np.sum(label_ratio)
        label_weight = np.square(1 / label_ratio)
        label_weight = label_weight.astype(np.float32)

    elif 'happi20' in args.dataset.lower():
        class Happi20Dataset(Dataset):
            def __init__(self, path, size, normalize=False):
                self.path = path
                self.size = size
                self.normalize = normalize
                self.precomp_mean = np.array([26.160023, 0.98314494, 0.116573125, -0.45998842, 0.1930554, 0.010749293, 98356.03, 100982.02, 216.13145, 258.9456, 3.765611e-08, 288.82578, 288.03925, 342.4827, 12031.449, 63.435772], dtype=np.float32)
                self.precomp_std =  np.array([17.04294, 8.164175, 5.6868863, 6.4967732, 5.4465833, 0.006383436, 7778.5957, 3846.1863, 9.791707, 14.35133, 1.8771327e-07, 19.866386, 19.094095, 624.22406, 679.5602, 4.2283397], dtype=np.float32)

            def __len__(self):
                return self.size

            def __getitem__(self, index):
                X = np.load(os.path.join(self.path, '%05d.x.npy' % index))
                if self.normalize:
                    X = (X - self.precomp_mean[None, :, None, None]) / self.precomp_std[None, :, None, None]
                Y = np.load(os.path.join(self.path, '%05d.y.npy' % index)).astype(np.int64)
                return X, Y

        dataset_train = Happi20Dataset(os.path.join(DATASET_PATH, 'train'), 43916, args.normalize_data)
        dataset_val = Happi20Dataset(os.path.join(DATASET_PATH, 'train'), 6274, args.normalize_data)
        dataset_test = Happi20Dataset(os.path.join(DATASET_PATH, 'train'), 12548, args.normalize_data)

        X_train = dataset_train.__getitem__(0)[0][None, ...]

        label_weight = np.array([0.00766805, 0.94184578, 0.05048618], np.float32)

    else:
        assert False, 'unknown segmentation dataset'

    if not args.use_weighted_loss:
        label_weight = np.mean(label_weight) * np.ones_like(label_weight)

    with gzip.open(os.path.join(DATASET_PATH, 'labels.gz'), 'rb') as f:
        class_names = pickle.load(f)[1:] # first class is <UNK>
        print('classes: %s' % ', '.join(class_names))

if args.task == 'classification':
    if 'smnist' in args.dataset.lower():
        VAL_SIZE = 10000
        num_classes = 10
        class_names = [str(i) for i in range(10)]

        with gzip.open(os.path.join(DATASET_PATH, 'train-images-idx3-ubyte.gz'), 'rb') as f:
            X = pickle.load(f).astype(np.float32) / 255
            X_train = X[:-VAL_SIZE]
            X_val = X[-VAL_SIZE:]

        with gzip.open(os.path.join(DATASET_PATH, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
            Y = pickle.load(f)
            Y_train = Y[:-VAL_SIZE]
            Y_val = Y[-VAL_SIZE:]

        with gzip.open(os.path.join(DATASET_PATH, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
            X_test = pickle.load(f).astype(np.float32) / 255

        with gzip.open(os.path.join(DATASET_PATH, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
            Y_test = pickle.load(f)

    else:
        assert False, 'unknown classification dataset'

if 'happi20' not in args.dataset.lower():
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    if args.normalize_data:
        with torch.no_grad():
            mu = torch.mean(X_train_tensor, dim=(0, 1, 3, 4), keepdims=True)
            sigma = torch.std(X_train_tensor, dim=(0, 1, 3, 4), keepdims=True)

            X_train_tensor = (X_train_tensor - mu) / sigma
            X_val_tensor = (X_val_tensor - mu) / sigma
            X_test_tensor = (X_test_tensor - mu) / sigma

    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
    dataset_val = TensorDataset(X_val_tensor, Y_val_tensor)
    dataset_test = TensorDataset(X_test_tensor, Y_test_tensor)

BATCH_SIZE_VAL = 500
BATCH_SIZE_TEST = 500

kwargs = {'num_workers': 8, 'pin_memory': True} if 'happi20' in args.dataset.lower() else {}
loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE_VAL, shuffle=False, **kwargs)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=False, **kwargs)

if args.task == 'segmentation':
    net = Network(net_body=SegmNet(
        in_channels=X_train.shape[2], in_width=X_train.shape[-1], num_classes=len(class_names),
        num_channels=args.num_channels, frac_hier=args.frac_hier, dropout_rate=args.dropout_rate,
        poly=args.polyhedron, model_type=args.model_type, oriented=args.oriented,
        bn=args.use_batch_norm, act=eval('nn.%s' % args.act_fn), pooling=args.pooling,
        attention=args.attention, cube_pad=args.use_cube_pad, label_weight=label_weight
    ))
elif args.task == 'classification':
    net = Network(net_body=ClassNet(
        in_channels=X_train.shape[2], in_width=X_train.shape[-1], num_classes=num_classes,
        num_channels=args.num_channels, frac_hier=args.frac_hier, dropout_rate=args.dropout_rate,
        poly=args.polyhedron, model_type=args.model_type, oriented=args.oriented,
        bn=args.use_batch_norm, act=eval('nn.%s' % args.act_fn), pooling=args.pooling,
        attention=args.attention, cube_pad=args.use_cube_pad
    ))
else:
    assert False, 'ERROR: unknown task `%s`' % args.task

# some tensors are created on first pass
with torch.no_grad():
    net(torch.tensor(X_train[:1]))

if args.use_dataparallel:
    net.net_body.seq = nn.DataParallel(net.net_body.seq)
DEVICE = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
net.to(DEVICE)
print(net)
print('%d parameters' % net.count_parameters())

if args.only_profile:
    import torchprof

    for (x, y) in dataset_val:
        x = x.to(DEVICE)
        with torchprof.Profile(net, use_cuda=args.gpu and torch.cuda.is_available()) as prof:
            net(x)
        break

    print(prof.display(show_events=False))
    print('***')
    print(prof.display(show_events=True))
    sys.exit(0)

opt = eval('optim.%s' % args.optimizer)(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_decay)
net.set_optimizer(opt)
sched = optim.lr_scheduler.StepLR(opt, step_size=args.sched_step_size, gamma=args.sched_gamma)


def test(net, loader_test, class_names, args):
    s = []
    stats_test = net.run(data_loader=loader_test, is_train=False)
    loss_test = stats_test['loss']
    acc_test = stats_test['acc']
    if args.task == 'segmentation':
        iou_test = stats_test['iou']
        prec_test = stats_test['prec']
        mAP_test = np.mean(stats_test['per_class_AP']) if 'happi20' not in args.dataset.lower() else np.mean(stats_test['per_class_AP'][1:])
        s.append('Test Loss: %12g | Test Acc: %9g | Test IoU: %9g | Test Prec: %9g | Test mAP: %9g' % (loss_test, acc_test, iou_test, prec_test, mAP_test))
        s.append('┌──────────┬───────── Per-Class Stats ─────────┬───────────┐')
        s.append('|  Class   |    Acc    |    Iou    |   Prec    |    AP     |')
        s.append('├──────────┼───────────┼───────────┼───────────┼───────────┤')
        for i, class_name in enumerate(class_names):
            s.append('| %8s | %9.7f | %9.7f | %9.7f | %9.7f |' % (
                class_name,
                stats_test['per_class_acc'][i],
                stats_test['per_class_iou'][i],
                stats_test['per_class_prec'][i],
                stats_test['per_class_AP'][i],
            ))
        s.append('└──────────┴───────────┴───────────┴───────────┴───────────┘')
    else:
        s.append('Test Loss: %12g | Test Acc: %9g' % (loss_test, acc_test))
    return '\n'.join(s)


checkpoint_file = os.path.join(args.checkpoint_path, 'checkpoint[%s].pt' % TAG)
if args.checkpoint_path is not None and os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['net_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict'])
    sched.load_state_dict(checkpoint['sched_state_dict'])
    best_state_dict = checkpoint['best_state_dict']
    t = checkpoint['t']
    t_mark_test = checkpoint['t_mark_test']
    t_mark_improve = checkpoint['t_mark_improve']
    best_score = checkpoint['best_score']
    best_acc_val = checkpoint['best_acc_val']
    best_iou_val = checkpoint['best_iou_val']
    best_prec_val = checkpoint['best_prec_val']
    best_mAP_val = checkpoint['best_mAP_val']
    log_lines = checkpoint['log_str'].split('\n')
    for (i, line) in enumerate(log_lines):
        if 'Training started' in line:
            print('\n'.join(log_lines[i:]), end='')
            break
    if t != args.num_epochs and t_mark_improve - t + 1 != args.num_epochs:
        print('[%s] Training resumed' % datetime.now())
else:
    assert args.num_epochs != 0
    print('[%s] Training started' % datetime.now())
    if args.num_epochs > 0:
        print('Training for %d epochs...' % args.num_epochs)
    else:
        print('Training with improvement threshold %d...' % -args.num_epochs)
    t = 0
    t_mark_test = 0
    t_mark_improve = -1
    best_score = -1e9
    best_state_dict, best_acc_val, best_iou_val, best_prec_val, best_mAP_val = None, None, None, None, None

while t != args.num_epochs and t_mark_improve - t + 1 != args.num_epochs:
    set_seed(1000 * (t + 1) + args.seed)
    stats_train = net.run(data_loader=loader_train, is_train=True)
    loss_train = stats_train['loss']
    acc_train = stats_train['acc']
    time_train = stats_train['time']
    if args.use_batch_norm:
        net.run(data_loader=loader_train, is_train=False, finetune=True)
    stats_val = net.run(data_loader=loader_val, is_train=False)
    loss_val = stats_val['loss']
    acc_val = stats_val['acc']
    if args.task == 'segmentation':
        iou_train = stats_train['iou']
        iou_val = stats_val['iou']
        prec_train = stats_train['prec']
        prec_val = stats_val['prec']
        mAP_train = np.mean(stats_train['per_class_AP']) if 'happi20' not in args.dataset.lower() else np.mean(stats_train['per_class_AP'][1:])
        mAP_val = np.mean(stats_val['per_class_AP']) if 'happi20' not in args.dataset.lower() else np.mean(stats_val['per_class_AP'][1:])
        print('Epoch: %3d | Train Loss: %12g | Train Acc: %9g | Train IoU: %9g | Train Prec: %9g | Train mAP: %9g | Val Loss: %12g | Val Acc: %9g | Val IoU: %9g | Val Prec: %9g | Val mAP: %9g | Train Time: %6.1f sec' % (
            t + 1, loss_train, acc_train, iou_train, prec_train, mAP_train, loss_val, acc_val, iou_val, prec_val, mAP_val, time_train), end='')
    else:
        print('Epoch: %3d | Train Loss: %12g | Train Acc: %9g | Val Loss: %12g | Val Acc: %9g | Train Time: %6.1f sec' % (
            t + 1, loss_train, acc_train, loss_val, acc_val, time_train), end='')
    if 'happi20' in args.dataset.lower():
        if args.use_weighted_loss:
            score = (acc_val + mAP_val) / 2
        else:
            score = acc_val * 0.1 + mAP_val * 0.9
    else:
        score = iou_val if args.task == 'segmentation' else acc_val
    print(' *' if score > best_score else '')
    if score > best_score:
        if t - t_mark_test >= 10:
            t_mark_test = t
            test_str = test(net, loader_test, class_names, args)
            test_str_obf = base64.b64encode(zlib.compress(test_str.encode())).decode()
            print('Test: %s' % test_str_obf)
        t_mark_improve = t
        best_score = score
        best_acc_val = acc_val
        if args.task == 'segmentation':
            best_iou_val = iou_val
            best_prec_val = prec_val
            best_mAP_val = mAP_val
        best_state_dict = net.state_dict()
    sched.step()
    t += 1
    torch.save({
        'net_state_dict': net.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'sched_state_dict': sched.state_dict(),
        'best_state_dict': best_state_dict,
        't': t,
        't_mark_test': t_mark_test,
        't_mark_improve': t_mark_improve,
        'best_score': best_score,
        'best_acc_val': best_acc_val,
        'best_iou_val': best_iou_val,
        'best_prec_val': best_prec_val,
        'best_mAP_val': best_mAP_val,
        'log_str': LOG_STR.getvalue()
    }, checkpoint_file)

print('Best Model Score: %9g' % best_score)
print('Best Model Val Acc: %9g' % best_acc_val)
if args.task == 'segmentation':
    print('Best Model Val IoU: %9g' % best_iou_val)
    print('Best Model Val Prec: %9g' % best_prec_val)
    print('Best Model Val mAP: %9g' % best_mAP_val)
print('[%s] Training finished' % datetime.now())

net.load_state_dict(best_state_dict)
net.save(os.path.join(args.base_path, 'model_best[%s].tar' % TAG), log=False)
test_str = test(net, loader_test, class_names, args)
test_str_obf = base64.b64encode(zlib.compress(test_str.encode())).decode()
print('Test: %s' % test_str_obf)

print(FINISH_TEXT)
LOG_STR.close()
LOG_FILE.close()
