import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from layers import *


class ClassNet(nn.Module):
    def __init__(self, in_channels, in_width, num_classes, num_channels, frac_hier, dropout_rate,
                 poly, model_type, oriented, bn, act, pooling, attention, cube_pad,
                 kernel_size=3):
        super(ClassNet, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.frac_hier = frac_hier
        self.dropout_rate = dropout_rate
        self.poly = poly
        self.model_type = model_type
        self.oriented = oriented
        self.bn = bn
        self.act = act
        self.pooling = pooling
        self.attention = attention
        self.cube_pad = (cube_pad and poly == 'cube')
        self.kernel_size = kernel_size
        self.num_faces = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[poly]

        assert poly in ['tetra', 'cube', 'octa', 'icosa']
        assert model_type in [1, 2]
        assert pooling in ['max', 'avg']
        assert kernel_size % 2 == 1
        if poly == 'cube':
            assert in_width % 8 == 0
        else:
            assert in_width % 8 == 1

        p = (kernel_size - 1) // 2
        num_orients = (4 if poly == 'cube' else 3)
        num_faces = self.num_faces
        ch = num_channels

        PxConvZ2 = (P4ConvZ2 if poly == 'cube' else P3ConvZ2)
        Conv = (P4ConvP4 if poly == 'cube' else P3ConvP3)
        if poly == 'cube':
            if pooling == 'max':
                Pool = lambda: MaxPool2d(kernel_size=2, stride=2)
            if pooling == 'avg':
                Pool = lambda: AvgPool2d(kernel_size=2, stride=2)
        else:
            if pooling == 'max':
                Pool = lambda: MaxPoolHex3(stride=2, padding=1)
            if pooling == 'avg':
                Pool = lambda: AvgPoolHex(kernel_size=3, stride=2, padding=1)

        w_list = []
        w = in_width
        for i in range(4):
            w_list.append(w)
            w = (w // 2 if poly == 'cube' else (w + 1) // 2)

        def HierSphereLayer(in_channels, out_channels, frac_hier, do_conv=True):
            return nn.Sequential(
                CubePad(p) if self.cube_pad else nn.Identity(),
                PoolPolyBroadcast(
                    face_conv=Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0) if do_conv else nn.Identity(),
                    att_conv_in=Conv(in_channels=int(in_channels * frac_hier), out_channels=int(out_channels * frac_hier), kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0) if attention else None,
                    att_conv_out=Conv(in_channels=int(in_channels * frac_hier), out_channels=int(out_channels * frac_hier), kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0) if attention else None,
                    poly=poly, model_type=model_type, oriented=oriented, aggr_fn='sum', in_channels=int(in_channels * frac_hier), out_channels=int(out_channels * frac_hier)
                )
            )

        self.seq = nn.Sequential(
            CubePad(p) if self.cube_pad else nn.Identity(),
            PxConvZ2(in_channels=in_channels, out_channels=ch, kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0),
            AssertShape(-1, num_faces, ch, num_orients, w_list[0], w_list[0]),
            PolyBatchNorm(poly=poly, num_channels=ch) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch, out_channels=ch, frac_hier=0),
            PolyBatchNorm(poly=poly, num_channels=ch) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch, out_channels=ch, frac_hier=frac_hier),

            Pool(),
            Conv(in_channels=ch, out_channels=ch * 2, kernel_size=1, stride=1),
            AssertShape(-1, num_faces, ch * 2, num_orients, w_list[1], w_list[1]),
            PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=0),
            PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=frac_hier),

            Pool(),
            Conv(in_channels=ch * 2, out_channels=ch * 4, kernel_size=1, stride=1),
            AssertShape(-1, num_faces, ch * 4, num_orients, w_list[2], w_list[2]),
            PolyBatchNorm(poly=poly, num_channels=ch * 4) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch * 4, out_channels=ch * 4, frac_hier=0),
            PolyBatchNorm(poly=poly, num_channels=ch * 4) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch * 4, out_channels=ch * 4, frac_hier=frac_hier),

            Pool(),
            Conv(in_channels=ch * 4, out_channels=ch * 8, kernel_size=1, stride=1),
            AssertShape(-1, num_faces, ch * 8, num_orients, w_list[3], w_list[3]),
            PolyBatchNorm(poly=poly, num_channels=ch * 8) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch * 8, out_channels=ch * 8, frac_hier=0),
            PolyBatchNorm(poly=poly, num_channels=ch * 8) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            HierSphereLayer(in_channels=ch * 8, out_channels=num_classes, frac_hier=1.0 if frac_hier > 0 else 0),

            GlobalPool(dim=[1, 3, 4, 5], pool_fn=torch.mean),
            Normalize(bias=0, scale=2 * w_list[3] / (1 + w_list[3])) if poly != 'cube' else nn.Identity(),
            AssertShape(-1, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        assert x.shape == (B, self.num_faces, self.in_channels, self.in_width, self.in_width), 'ERROR: Unexpected data shape. Maybe dataset does not match model.'
        x = self.seq(x)
        assert x.shape == (B, self.num_classes)
        return x

    def loss_fn(self, y_logits, y_true):
        B = y_logits.shape[0]
        assert y_logits.shape == (B, self.num_classes)
        assert y_true.shape == (B,)
        loss = F.cross_entropy(input=y_logits, target=y_true, reduction='none')
        assert loss.shape == (B,)
        return loss

    def init_stats(self):
        return {'acc': 0, 'cnt': 0}

    def update_stats(self, x, y, y_, stats):
        with torch.no_grad():
            stats['acc'] += torch.sum(100.0 * (torch.argmax(y_, dim=1) == y)).cpu().item()
            stats['cnt'] += x.shape[0]
        return stats

    def finalize_stats(self, stats):
        return {'acc': stats['acc'] / stats['cnt']}


class SegmNet(nn.Module):
    def __init__(self, in_channels, in_width, num_classes, num_channels, frac_hier, dropout_rate,
                 poly, model_type, oriented, bn, act, pooling, attention, cube_pad,
                 label_weight=None, kernel_size=3):
        super(SegmNet, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.frac_hier = frac_hier
        self.dropout_rate = dropout_rate
        self.poly = poly
        self.model_type = model_type
        self.oriented = oriented
        self.bn = bn
        self.act = act
        self.pooling = pooling
        self.attention = attention
        self.cube_pad = (cube_pad and poly == 'cube')
        if label_weight is not None:
            self.register_buffer('label_weight', torch.tensor(label_weight))
        else:
            self.label_weight = None
        self.kernel_size = kernel_size
        self.num_faces = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[poly]

        assert poly in ['tetra', 'cube', 'octa', 'icosa']
        assert model_type in [1, 2]
        assert pooling in ['max', 'avg']
        assert kernel_size % 2 == 1
        if poly == 'cube':
            assert in_width % 8 == 0
        else:
            assert in_width % 8 == 1

        p = (kernel_size - 1) // 2
        num_orients = (4 if poly == 'cube' else 3)
        num_faces = self.num_faces
        ch = num_channels

        PxConvZ2 = (P4ConvZ2 if poly == 'cube' else P3ConvZ2)
        Conv = (P4ConvP4 if poly == 'cube' else P3ConvP3)
        if poly == 'cube':
            if pooling == 'max':
                Pool = lambda: MaxPool2d(kernel_size=2, stride=2)
            if pooling == 'avg':
                Pool = lambda: AvgPool2d(kernel_size=2, stride=2)
            Unpool = lambda: Interpolate(scale_factor=2, mode='area')
        else:
            if pooling == 'max':
                Pool = lambda: MaxPoolHex3(stride=2, padding=1)
            if pooling == 'avg':
                Pool = lambda: AvgPoolHex(kernel_size=3, stride=2, padding=1)
            Unpool = lambda: UnpoolHex()

        w_list = []
        w = in_width
        for i in range(4):
            w_list.append(w)
            w = (w // 2 if poly == 'cube' else (w + 1) // 2)

        def HierSphereLayer(in_channels, out_channels, frac_hier, do_conv=True):
            return nn.Sequential(
                CubePad(p) if self.cube_pad else nn.Identity(),
                PoolPolyBroadcast(
                    face_conv=Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0) if do_conv else nn.Identity(),
                    att_conv_in=Conv(in_channels=int(in_channels * frac_hier), out_channels=int(out_channels * frac_hier), kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0) if attention else None,
                    att_conv_out=Conv(in_channels=int(in_channels * frac_hier), out_channels=int(out_channels * frac_hier), kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0) if attention else None,
                    poly=poly, model_type=model_type, oriented=oriented, aggr_fn='sum', in_channels=int(in_channels * frac_hier), out_channels=int(out_channels * frac_hier),
                )
            )

        self.seq = nn.Sequential(
            UBlock(cat_dim=2, path=nn.Sequential(
                CubePad(p) if self.cube_pad else nn.Identity(),
                PxConvZ2(in_channels=in_channels, out_channels=ch, kernel_size=kernel_size, stride=1, padding=p if not self.cube_pad else 0),
                AssertShape(-1, num_faces, ch, num_orients, w_list[0], w_list[0]),
                PolyBatchNorm(poly=poly, num_channels=ch) if bn else nn.Identity(),
                act(),
                nn.Dropout(p=dropout_rate),
                HierSphereLayer(in_channels=ch, out_channels=ch, frac_hier=0),
                PolyBatchNorm(poly=poly, num_channels=ch) if bn else nn.Identity(),
                act(),
                nn.Dropout(p=dropout_rate),
                HierSphereLayer(in_channels=ch, out_channels=ch, frac_hier=frac_hier),
                UBlock(cat_dim=2, path=nn.Sequential(
                    Pool(),
                    Conv(in_channels=ch, out_channels=ch * 2, kernel_size=1, stride=1),
                    AssertShape(-1, num_faces, ch * 2, num_orients, w_list[1], w_list[1]),
                    PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
                    act(),
                    nn.Dropout(p=dropout_rate),
                    HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=0),
                    PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
                    act(),
                    nn.Dropout(p=dropout_rate),
                    HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=frac_hier),
                    UBlock(cat_dim=2, path=nn.Sequential(
                        Pool(),
                        Conv(in_channels=ch * 2, out_channels=ch * 4, kernel_size=1, stride=1),
                        AssertShape(-1, num_faces, ch * 4, num_orients, w_list[2], w_list[2]),
                        PolyBatchNorm(poly=poly, num_channels=ch * 4) if bn else nn.Identity(),
                        act(),
                        nn.Dropout(p=dropout_rate),
                        HierSphereLayer(in_channels=ch * 4, out_channels=ch * 4, frac_hier=0),
                        PolyBatchNorm(poly=poly, num_channels=ch * 4) if bn else nn.Identity(),
                        act(),
                        nn.Dropout(p=dropout_rate),
                        HierSphereLayer(in_channels=ch * 4, out_channels=ch * 4, frac_hier=frac_hier),
                        UBlock(cat_dim=2, path=nn.Sequential(
                            Pool(),
                            Conv(in_channels=ch * 4, out_channels=ch * 8, kernel_size=1, stride=1),
                            AssertShape(-1, num_faces, ch * 8, num_orients, w_list[3], w_list[3]),
                            PolyBatchNorm(poly=poly, num_channels=ch * 8) if bn else nn.Identity(),
                            act(),
                            nn.Dropout(p=dropout_rate),
                            HierSphereLayer(in_channels=ch * 8, out_channels=ch * 8, frac_hier=0),
                            PolyBatchNorm(poly=poly, num_channels=ch * 8) if bn else nn.Identity(),
                            act(),
                            nn.Dropout(p=dropout_rate),
                            HierSphereLayer(in_channels=ch * 8, out_channels=ch * 8, frac_hier=frac_hier),
                            Unpool(),
                            Conv(in_channels=ch * 8, out_channels=ch * 4, kernel_size=1, stride=1)
                        )),
                        AssertShape(-1, num_faces, ch * 8, num_orients, w_list[2], w_list[2]),
                        PolyBatchNorm(poly=poly, num_channels=ch * 8) if bn else nn.Identity(),
                        act(),
                        nn.Dropout(p=dropout_rate),
                        HierSphereLayer(in_channels=ch * 8, out_channels=ch * 4, frac_hier=0),
                        PolyBatchNorm(poly=poly, num_channels=ch * 4) if bn else nn.Identity(),
                        act(),
                        nn.Dropout(p=dropout_rate),
                        HierSphereLayer(in_channels=ch * 4, out_channels=ch * 4, frac_hier=frac_hier),
                        Unpool(),
                        Conv(in_channels=ch * 4, out_channels=ch * 2, kernel_size=1, stride=1)
                    )),
                    AssertShape(-1, num_faces, ch * 4, num_orients, w_list[1], w_list[1]),
                    PolyBatchNorm(poly=poly, num_channels=ch * 4) if bn else nn.Identity(),
                    act(),
                    nn.Dropout(p=dropout_rate),
                    HierSphereLayer(in_channels=ch * 4, out_channels=ch * 2, frac_hier=0),
                    PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
                    act(),
                    nn.Dropout(p=dropout_rate),
                    HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=frac_hier),
                    Unpool(),
                    Conv(in_channels=ch * 2, out_channels=ch, kernel_size=1, stride=1)
                )),
                AssertShape(-1, num_faces, ch * 2, num_orients, w_list[0], w_list[0]),
                PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
                act(),
                nn.Dropout(p=dropout_rate),
                HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=0),
                PolyBatchNorm(poly=poly, num_channels=ch * 2) if bn else nn.Identity(),
                act(),
                nn.Dropout(p=dropout_rate),
                HierSphereLayer(in_channels=ch * 2, out_channels=ch * 2, frac_hier=frac_hier),
                GlobalPool(dim=[3], pool_fn=torch.mean)
            )),
            PolyBatchNorm(poly=poly, num_channels=ch * 2 + in_channels) if bn else nn.Identity(),
            act(),
            nn.Dropout(p=dropout_rate),
            PxConvZ2(in_channels=ch * 2 + in_channels, out_channels=num_classes, kernel_size=1, stride=1),
            GlobalPool(dim=[3], pool_fn=torch.mean),
            AssertShape(-1, num_faces, num_classes, in_width, in_width)
        )

    def forward(self, x):
        B = x.shape[0]
        assert x.shape == (B, self.num_faces, self.in_channels, self.in_width, self.in_width)
        x = self.seq(x)
        assert x.shape == (B, self.num_faces, self.num_classes, self.in_width, self.in_width)
        return x

    def loss_fn(self, y_logits, y_true):
        B = y_logits.shape[0]
        assert y_logits.shape == (B, self.num_faces, self.num_classes, self.in_width, self.in_width)
        assert y_true.shape == (B, self.num_faces, self.in_width, self.in_width)
        y_logits = y_logits.view(B * self.num_faces, self.num_classes, self.in_width, self.in_width)
        y_true = y_true.view(B * self.num_faces, self.in_width, self.in_width)
        mask = (y_true != 0) * 1.0
        y_true = torch.where(y_true != 0, y_true - 1, y_true)
        loss = F.cross_entropy(input=y_logits, target=y_true, weight=self.label_weight, reduction='none') * mask
        assert loss.shape == (B * self.num_faces, self.in_width, self.in_width)
        loss = loss.view(B, self.num_faces, self.in_width, self.in_width)
        loss = torch.mean(loss, dim=[1, 2, 3])
        assert loss.shape == (B,)
        return loss

    def init_stats(self):
        return {
            'tp': [0] * self.num_classes,
            'fp': [0] * self.num_classes,
            'tn': [0] * self.num_classes,
            'fn': [0] * self.num_classes,

            'y': [],
            'y_': [],
            'mask': []
        }

    def update_stats(self, x, y, y_, stats):
        with torch.no_grad():
            mask = 1 * (y != 0)
            stats['mask'].append(mask.cpu().numpy())
            y = torch.where(y != 0, y - 1, y)
            stats['y'].append(y.cpu().numpy())
            stats['y_'].append(y_.cpu().numpy())
            y_ = torch.argmax(y_, dim=2)
            for i in range(self.num_classes):
                stats['tp'][i] += torch.sum(((1 * (y_ == i) + (y == i)) == 2) * mask).cpu().item()
                stats['fp'][i] += torch.sum(((1 * (y_ == i) + (y != i)) == 2) * mask).cpu().item()
                stats['tn'][i] += torch.sum(((1 * (y_ != i) + (y != i)) == 2) * mask).cpu().item()
                stats['fn'][i] += torch.sum(((1 * (y_ != i) + (y == i)) == 2) * mask).cpu().item()
        return stats

    def finalize_stats(self, stats):
        stats['per_class_acc'] = [stats['tp'][i] / (stats['tp'][i] + stats['fn'][i] + 1e-10) for i in range(self.num_classes)]
        stats['per_class_iou'] = [stats['tp'][i] / (stats['tp'][i] + stats['fn'][i] + stats['fp'][i] + 1e-10) for i in range(self.num_classes)]
        stats['per_class_prec'] = [stats['tp'][i] / (stats['tp'][i] + stats['fp'][i] + 1e-10) for i in range(self.num_classes)]
        stats['acc'] = sum(stats['per_class_acc']) / self.num_classes
        stats['iou'] = sum(stats['per_class_iou']) / self.num_classes
        stats['prec'] = sum(stats['per_class_prec']) / self.num_classes

        stats['y'] = label_binarize(np.concatenate(stats['y']).reshape(-1), classes=np.arange(self.num_classes))
        stats['y_'] = np.concatenate(stats['y_']).transpose(0, 1, 3, 4, 2).reshape(-1, self.num_classes).round(2)
        stats['mask'] = np.concatenate(stats['mask']).reshape(-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stats['per_class_AP'] = average_precision_score(stats['y'], stats['y_'], average=None, sample_weight=stats['mask'])
        return stats


def bn_reset_running_stats(net):
    net.apply(lambda x: x.reset_running_stats() if isinstance(x, nn.BatchNorm1d) else None)


def bn_train_mode(net):
    net.apply(lambda x: x.train() if isinstance(x, nn.BatchNorm1d) else None)


class Network(nn.Module):
    ''' Wraps around a neural network and adds methods such as `save`, `load`, and `run`. '''
    def __init__(self, net_body):
        super(Network, self).__init__()
        self.net_body = net_body

    def save(self, path, log=True):
        torch.save(self.state_dict(), path)
        if log:
            print('Saved model to `%s`' % path)

    def load(self, path, log=True, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))
        if log:
            print('Loaded model from `%s`' % path)

    def set_optimizer(self, opt):
        self.opt = opt

    def get_device(self):
        '''
        Returns the `torch.device` on which the network resides.
        This method only makes sense when all module parameters reside on the **same** device.
        '''
        return list(self.parameters())[0].device

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x):
        return self.net_body(x)

    def run(self, data_loader, is_train, finetune=False, opt=None):
        ''' This method handles training and evaluation. '''
        assert not (finetune and is_train)
        device = self.get_device()
        if is_train:
            opt = opt or self.opt
            assert opt
            bn_reset_running_stats(self.net_body) # assuming momentum=None
            self.train()
        else:
            self.eval()
            if finetune:
                bn_reset_running_stats(self.net_body)
                bn_train_mode(self.net_body)
        m = 0
        loss = 0
        stats = self.net_body.init_stats()
        time_start = time.time()
        with torch.set_grad_enabled(is_train):
            for (x, y) in data_loader:
                x = x.to(device)
                y = y.to(device)
                m += x.shape[0]
                y_ = self(x)
                batch_loss = self.net_body.loss_fn(y_, y)
                if is_train:
                    opt.zero_grad()
                    torch.mean(batch_loss).backward()
                    opt.step()
                stats = self.net_body.update_stats(x, y, y_, stats)
                loss += torch.sum(batch_loss.detach()).cpu().item()
        time_end = time.time()
        stats = self.net_body.finalize_stats(stats)
        stats['loss'] = loss / m
        stats['time'] = time_end - time_start
        return stats
