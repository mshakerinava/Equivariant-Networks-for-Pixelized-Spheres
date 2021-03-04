import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from groupy.hexa import mask
import groupy.gconv.pytorch_gconv as gconv
from autoequiv import *
from symmetry import *


class Patch:
    @staticmethod
    def max(x, dim):
        if type(dim) != list:
            return torch.max(x, dim)[0]
        dim.sort()
        for i in dim[::-1]:
            x = torch.max(x, dim=i)[0]
        return x


class P3ConvZ2(gconv.P3ConvZ2):
    def __init__(self, *args, **kwargs):
        super(P3ConvZ2, self).__init__(*args, **kwargs, image_shape='triangle')

    def forward(self, x):
        shape_bac = x.shape[:-3]
        x = x.view(np.prod(shape_bac), *x.shape[-3:])
        x = super(P3ConvZ2, self).forward(x)
        x = x.view(*shape_bac, *x.shape[1:])
        return x

    def __repr__(self):
        return 'P3ConvZ2(in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s, padding=%s, bias=%r)' % (
            self.in_channels, self.out_channels, str(self.kernel_size), str(self.stride), str(self.padding), self.bias is not None)


class P3ConvP3(gconv.P3ConvP3):
    def __init__(self, *args, **kwargs):
        super(P3ConvP3, self).__init__(*args, **kwargs, image_shape='triangle')

    def forward(self, x):
        assert x.shape[-3] == 3
        shape_bac = x.shape[:-4]
        x = x.view(np.prod(shape_bac), *x.shape[-4:])
        x = super(P3ConvP3, self).forward(x)
        x = x.view(*shape_bac, *x.shape[1:])
        return x

    def __repr__(self):
        return 'P3ConvP3(in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s, padding=%s, bias=%r)' % (
            self.in_channels, self.out_channels, str(self.kernel_size), str(self.stride), str(self.padding), self.bias is not None)


class P4ConvZ2(gconv.P4ConvZ2):
    def __init__(self, *args, **kwargs):
        super(P4ConvZ2, self).__init__(*args, **kwargs)

    def forward(self, x):
        shape_bac = x.shape[:-3]
        x = x.view(np.prod(shape_bac), *x.shape[-3:])
        x = super(P4ConvZ2, self).forward(x)
        x = x.view(*shape_bac, *x.shape[1:])
        return x

    def __repr__(self):
        return 'P4ConvZ2(in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s, padding=%s, bias=%r)' % (
            self.in_channels, self.out_channels, str(self.kernel_size), str(self.stride), str(self.padding), self.bias is not None)


class P4ConvP4(gconv.P4ConvP4):
    def __init__(self, *args, **kwargs):
        super(P4ConvP4, self).__init__(*args, **kwargs)

    def forward(self, x):
        assert x.shape[-3] == 4
        shape_bac = x.shape[:-4]
        x = x.view(np.prod(shape_bac), *x.shape[-4:])
        x = super(P4ConvP4, self).forward(x)
        x = x.view(*shape_bac, *x.shape[1:])
        return x

    def __repr__(self):
        return 'P4ConvP4(in_channels=%d, out_channels=%d, kernel_size=%s, stride=%s, padding=%s, bias=%r)' % (
            self.in_channels, self.out_channels, str(self.kernel_size), str(self.stride), str(self.padding), self.bias is not None)


class BatchNorm(nn.Module):
    def __init__(self, num_channels, momentum=None, eps=2e-5):
        super(BatchNorm, self).__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.bn = nn.BatchNorm1d(num_channels, momentum=momentum, eps=eps)

    def forward(self, x):
        # DOC: input is assumed to have shape (B, `num_channels`, ...)
        B = x.shape[0]
        shape_bac = x.shape
        x = x.view(B, self.num_channels, -1).contiguous()
        x = self.bn(x)
        x = x.view(shape_bac).contiguous()
        return x

    def __repr__(self):
        return 'BatchNorm(num_chanels=%d, momentum=%g, eps=%g)' % (
            self.num_channels, self.momentum if self.momentum is not None else -1, self.eps)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert len(x.shape) >= 4
        shape_bac = x.shape[1:-2]
        x = x.view(x.shape[0], np.prod(x.shape[1:-2]), *x.shape[-2:])
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        x = x.view(x.shape[0], *shape_bac, *x.shape[-2:])
        return x

    def __repr__(self):
        return 'MaxPool2d(kernel_size=%s, stride=%s, padding=%s)' % (
            str(self.kernel_size), str(self.stride), str(self.padding))


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert len(x.shape) >= 4
        shape_bac = x.shape[1:-2]
        x = x.view(x.shape[0], np.prod(x.shape[1:-2]), *x.shape[-2:])
        x = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        x = x.view(x.shape[0], *shape_bac, *x.shape[-2:])
        return x

    def __repr__(self):
        return 'AvgPool2d(kernel_size=%s, stride=%s, padding=%s)' % (
            str(self.kernel_size), str(self.stride), str(self.padding))


class _MaxPoolHex(nn.Module):
    def __init__(self, kernel_size, stride, padding, pad_value=1e-9):
        super(_MaxPoolHex, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pad_value = pad_value
        hex_mask = mask.hexagon_axial(self.kernel_size)
        size = int(np.sum(hex_mask))
        weight = np.zeros((size, kernel_size, kernel_size))
        cnt = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if hex_mask[i, j] == 1:
                    weight[cnt, i, j] = 1
                    cnt += 1
        weight = weight[:, None, :, :]
        self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))

    def forward(self, x):
        assert len(x.shape) >= 2
        assert x.shape[-1] == x.shape[-2]
        shape_bac = x.shape[:-2]
        x = x.view(-1, 1, *x.shape[-2:])
        w = x.shape[-1]
        x[:, :, torch.arange(w)[None, :] == w - torch.arange(w)[:, None]] = self.pad_value # assuming triangle shaped input
        x = F.pad(x, (self.padding,) * 4, value=self.pad_value)
        x = F.conv2d(x, weight=self.weight, bias=None, stride=self.stride, padding=0)
        x = torch.max(x, dim=1)[0]
        if not hasattr(self, 'output_mask'):
            output_mask = mask.triangle_axial(*x.shape[-2:])
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.bool, device=x.device))
        x = x * self.output_mask
        x = x.view(*shape_bac, *x.shape[-2:])
        return x

    def __repr__(self):
        return 'MaxPoolHex(kernel_size=%s, stride=%s, padding=%s)' % (
            str(self.kernel_size), str(self.stride), str(self.padding))


class MaxPoolHex3(nn.Module):
    # DOC: kernel size is 3
    def __init__(self, stride, padding, pad_value=-1e9):
        super(MaxPoolHex3, self).__init__()
        self.stride = stride
        self.padding = padding
        self.pad_value = pad_value

    def forward(self, x):
        assert len(x.shape) >= 4
        shape_bac = x.shape[1:-2]
        x = x.view(x.shape[0], np.prod(x.shape[1:-2]), *x.shape[-2:])
        w = x.shape[-1]
        x[:, :, torch.arange(w)[None, :] == w - torch.arange(w)[:, None]] = self.pad_value # assuming triangle shaped input
        x = F.pad(x, (self.padding,) * 4, value=self.pad_value)
        x1 = F.max_pool2d(x[:, :, 1:, :-1], 2, self.stride)
        x2 = F.max_pool2d(x[:, :, :-1, 1:], 2, self.stride)
        x = torch.max(x1, x2)
        if not hasattr(self, 'output_mask'):
            output_mask = mask.triangle_axial(*x.shape[-2:])
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.bool, device=x.device))
        x = x * self.output_mask
        x = x.view(x.shape[0], *shape_bac, *x.shape[-2:])
        return x

    def __repr__(self):
        return 'MaxPoolHex3(stride=%s, padding=%s)' % (
            str(self.stride), str(self.padding))


class AvgPoolHex(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(AvgPoolHex, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        weight = mask.hexagon_axial(self.kernel_size)
        weight /= np.sum(weight)
        weight = weight[None, None, :, :]
        self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))

    def forward(self, x):
        assert len(x.shape) >= 4
        shape_bac = x.shape[:-2]
        x = x.view(-1, 1, *x.shape[-2:])
        x = F.conv2d(x, weight=self.weight, bias=None, stride=self.stride, padding=self.padding)
        if not hasattr(self, 'output_mask'):
            output_mask = mask.triangle_axial(*x.shape[-2:])
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.bool, device=x.device))
        x = x * self.output_mask
        x = x.view(*shape_bac, *x.shape[-2:])
        return x

    def __repr__(self):
        return 'AvgPoolHex(kernel_size=%s, stride=%s, padding=%s)' % (
            str(self.kernel_size), str(self.stride), str(self.padding))


class UnpoolHex(nn.Module):
    def __init__(self, mode='area'):
        super(UnpoolHex, self).__init__()
        self.mode = mode

    def forward(self, x):
        assert len(x.shape) >= 4
        shape_bac = x.shape[1:-2]
        x = x.view(x.shape[0], np.prod(x.shape[1:-2]), *x.shape[-2:])
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        x1 = F.pad(x, (1, 0, 0, 1), value=0)
        x2 = F.pad(x, (0, 1, 1, 0), value=0)
        x = (x1 + x2) / 2
        x = x[:, :, 1:-1, 1:-1]
        if not hasattr(self, 'output_mask'):
            output_mask = mask.triangle_axial(*x.shape[-2:])
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.bool, device=x.device))
        x = x * self.output_mask
        x = x.view(x.shape[0], *shape_bac, *x.shape[-2:])
        return x


class GlobalPool(nn.Module):
    def __init__(self, dim, pool_fn):
        super(GlobalPool, self).__init__()
        self.dim = [dim] if type(dim) == int else dim
        self.pool_fn = pool_fn

    def forward(self, x):
        return self.pool_fn(x, dim=self.dim)

    def __repr__(self):
        return 'GlobalPool(dim=%s, pool_fn=%s)' % (
            str(self.dim), self.pool_fn.__name__)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()

    def __repr__(self):
        return 'Transpose(dim0=%d, dim1=%d)' % (self.dim0, self.dim1)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        shape_bac = x.shape
        x = x.view(x.shape[0], np.prod(x.shape[1: -2]), *x.shape[-2:])
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = x.view(*shape_bac[:-2], *(int(w * self.scale_factor) for w in shape_bac[-2:]))
        return x

    def __repr__(self):
        return 'Interpolate(scale_factor=%g, mode=%s)' % (self.scale_factor, self.mode)


class AssertShape(nn.Module):
    def __init__(self, *shape):
        super(AssertShape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # print('ASSERT_SHAPE(x.shape = %s, self.shape = %s)' % (str(x.shape), str(self.shape)))
        assert len(x.shape) == len(self.shape)
        for i in range(len(self.shape)):
            if self.shape[i] != -1:
                assert x.shape[i] == self.shape[i]
        return x

    def __repr__(self):
        return 'AssertShape(%s)' % (', '.join(['%d' % x for x in self.shape]))


class ResBlock(nn.Module):
    def __init__(self, main_path, shortcut_path=lambda x: x):
        super(ResBlock, self).__init__()
        self.main_path = main_path
        self.shortcut_path = shortcut_path

    def forward(self, x):
        x_ = self.main_path(x)
        x = self.shortcut_path(x)
        assert x.shape == x_.shape
        y = x + x_
        return y


class UBlock(nn.Module):
    def __init__(self, cat_dim, path):
        super(UBlock, self).__init__()
        self.cat_dim = cat_dim
        self.path = path

    def forward(self, x):
        y = self.path(x)
        x_shape, y_shape = list(x.shape), list(y.shape)
        # the size along `cat_dim` does not need to match
        y_shape[self.cat_dim] = x_shape[self.cat_dim] = 0
        assert x_shape == y_shape
        return torch.cat([x, y], dim=self.cat_dim)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

    def __repr__(self):
        return 'View(%s)' % (', '.join(['%d' % x for x in self.shape]))


class Normalize(nn.Module):
    def __init__(self, bias, scale):
        super(Normalize, self).__init__()
        self.register_buffer('bias', torch.tensor(bias, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return (x - self.bias) / self.scale

    def __repr__(self):
        return 'Normalize(bias=%g, scale=%g)' % (self.bias.item(), self.scale.item())


class Parallel(nn.Module):
    def __init__(self, *layers):
        super(Parallel, self).__init__()
        self.layers = [x for x in layers if x is not None]

    def forward(self, x):
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y += layer(x)
        return y


def poly_layer_generators(poly, model_type, oriented):
    rot1_z = symmetry_perm(poly, model_type=1, around_z=True)
    rot1_o = symmetry_perm(poly, model_type=1, around_z=False)

    rot2_z = symmetry_perm(poly, model_type=2, around_z=True)
    rot2_o = symmetry_perm(poly, model_type=2, around_z=False)

    if model_type == 1:
        in_generators = [rot1_o, rot1_z] if not oriented else [rot1_z]
        out_generators = in_generators
    elif model_type == 2:
        in_generators = [rot2_o, rot2_z] if not oriented else [rot2_z]
        out_generators = in_generators
    else:
        assert False

    return in_generators, out_generators


class PolyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, poly, model_type, oriented):
        super(PolyLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.poly = poly
        self.model_type = model_type
        self.oriented = oriented
        in_generators, out_generators = poly_layer_generators(poly, model_type, oriented)
        self.lin_equiv = LinearEquiv(in_generators, out_generators, in_channels, out_channels)

    def forward(self, x):
        return self.lin_equiv(x)

    def __repr__(self):
        return 'PolyLayer(in_channels=%d, out_channels=%d, poly=%s, model_type=%d, oriented=%r)' % (
            self.in_channels, self.out_channels, self.poly, self.model_type, self.oriented)


class PoolPolyBroadcast(nn.Module):
    def __init__(self, face_conv, att_conv_in, att_conv_out,
                 poly, model_type, oriented, aggr_fn='sum',
                 in_channels=None, out_channels=None,
                 pool_fn=torch.mean):
        super(PoolPolyBroadcast, self).__init__()
        self.face_conv = face_conv
        self.poly = poly
        self.num_faces = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[poly]
        self.num_orients = 4 if poly == 'cube' else 3
        self.model_type = model_type
        self.oriented = oriented
        self.aggr_fn = aggr_fn
        assert aggr_fn in ['sum', 'cat']
        self.att_conv_in = att_conv_in
        if att_conv_in is not None:
            assert att_conv_in.in_channels <= face_conv.in_channels
        self.att_conv_out = att_conv_out
        if att_conv_out is not None:
            assert att_conv_out.in_channels <= face_conv.in_channels
        self.in_channels = in_channels if in_channels != None else face_conv.in_channels
        self.out_channels = out_channels if out_channels != None else face_conv.out_channels
        assert self.in_channels <= face_conv.in_channels
        assert self.out_channels <= face_conv.out_channels
        self.pool_fn = pool_fn
        if poly != 'cube' and pool_fn.__name__ != 'mean':
            print('WARNING: unsupported pooling operation.')
            # e.g., `max` will produce zero if all values are negative
        if self.in_channels != 0 and self.out_channels != 0:
            self.poly_layer = PolyLayer(self.in_channels, self.out_channels, poly, model_type, oriented)
        else:
            self.poly_layer = None

    def forward(self, x):
        y = self.face_conv(x)
        if self.poly_layer == None:
            return y

        if self.poly != 'cube' and not hasattr(self, 'input_mask'):
            input_mask = mask.triangle_axial(*x.shape[-2:])
            self.register_buffer('input_mask', torch.tensor(input_mask, dtype=torch.bool, device=x.device))

        if self.poly != 'cube' and not hasattr(self, 'output_mask'):
            output_mask = mask.triangle_axial(*y.shape[-2:])
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.bool, device=y.device))

        if self.att_conv_in is not None:
            att_mask_in = self.att_conv_in(x[:, :, :self.att_conv_in.in_channels])
            if self.poly != 'cube':
                att_mask_in = torch.where(self.input_mask, att_mask_in, -1e9)
            shape_bac = att_mask_in.shape
            # TODO: try without softmax
            att_mask_in = F.softmax(att_mask_in.view(*shape_bac[:-2], -1), dim=-1).view(shape_bac)
            att_mask_in = att_mask_in * np.prod(shape_bac[-2:])
            x_shape = list(x.shape)
            x_shape[2] = self.in_channels
            assert list(att_mask_in.shape) == x_shape
        else:
            att_mask_in = 1

        if self.att_conv_out is not None:
            att_mask_out = self.att_conv_out(x[:, :, :self.att_conv_out.in_channels])
            if self.poly != 'cube':
                att_mask_out = torch.where(self.input_mask, att_mask_out, -1e9)
            shape_bac = att_mask_out.shape
            # TODO: try without softmax
            att_mask_out = F.softmax(att_mask_out.view(*shape_bac[:-2], -1), dim=-1).view(shape_bac)
            att_mask_out = att_mask_out * np.prod(shape_bac[-2:])
            y_shape = list(y.shape)
            y_shape[2] = self.out_channels
            assert list(att_mask_out.shape) == y_shape
        else:
            att_mask_out = 1

        B = x.shape[0]
        ch_in = x.shape[2]
        ch_out = y.shape[2]
        w = x.shape[4]
        assert x.shape == (B, self.num_faces, ch_in, self.num_orients, w, w)
        x = x[:, :, :self.in_channels] * att_mask_in
        x = torch.transpose(x, 1, 2).contiguous()

        if self.model_type == 1:
            x = self.pool_fn(x, dim=[3, 4, 5])
        elif self.model_type == 2:
            x = self.pool_fn(x, dim=[4, 5])
            x = x.view(B, self.in_channels, self.num_faces * self.num_orients)
        else:
            assert False

        if self.poly != 'cube' and self.pool_fn.__name__ == 'mean':
            # each face is a triangle so normalization must be adjusted!
            if not hasattr(self, 'renorm_factor'):
                renorm_factor = 2 * w / (w + 1)
                self.register_buffer('renorm_factor', torch.tensor(renorm_factor, dtype=torch.float32, device=x.device))
            x = x * self.renorm_factor

        x = self.poly_layer(x)

        if self.model_type == 1:
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(B, self.num_faces, self.out_channels, 1, 1, 1)
        elif self.model_type == 2:
            x = x.view(B, self.out_channels, self.num_faces, self.num_orients)
            x = torch.transpose(x, 1, 2)
            x = x.view(B, self.num_faces, self.out_channels, self.num_orients, 1, 1)
        else:
            assert False

        x = x.expand(-1, -1, -1, self.num_orients, y.shape[-2], y.shape[-1]) * att_mask_out
        if self.aggr_fn == 'sum':
            y[:, :, :self.out_channels] += x
        elif self.aggr_fn == 'cat':
            y = torch.cat((x, y), dim=2)
        else:
            assert False

        if self.poly != 'cube':
            y = y * self.output_mask
        return y


class PolyBatchNorm(nn.Module):
    def __init__(self, poly, num_channels):
        super(PolyBatchNorm, self).__init__()
        self.poly = poly
        self.num_channels = num_channels
        self.bn = BatchNorm(num_channels)
        self.num_faces = {'icosa': 20, 'octa': 8, 'cube': 6, 'tetra': 4}[poly]

    def forward(self, x):
        assert x.shape[1] == self.num_faces
        x = torch.transpose(x, 1, 2).contiguous()
        x = self.bn(x)
        x = torch.transpose(x, 1, 2).contiguous()
        if self.poly != 'cube' and not hasattr(self, 'output_mask'):
            output_mask = mask.triangle_axial(*x.shape[-2:])
            self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.bool, device=x.device))
        if self.poly != 'cube':
            x = x * self.output_mask
        return x

    def __repr__(self):
        return 'PolyBatchNorm(poly=%s, num_channels=%d)' % (self.poly, self.num_channels)


class CubePad(nn.Module):
    def __init__(self, padding):
        super(CubePad, self).__init__()
        self.padding = padding

    @staticmethod
    def square_pad(x, le, up, ri, do, pad):
        assert pad >= 1
        shape_bac = x.shape
        x, le, up, ri, do = [y.reshape(-1, *y.shape[-2:]) for y in [x, le, up, ri, do]]
        x = F.pad(x, pad=(pad, pad, pad, pad))
        x[:, pad:-pad, :pad] = le[:, :, -pad:]
        x[:, :pad, pad:-pad] = up[:, -pad:, :]
        x[:, pad:-pad, -pad:] = ri[:, :, :pad]
        x[:, -pad:, pad:-pad] = do[:, :pad, :]
        x = x.view(*shape_bac[:-2], *x.shape[-2:])
        return x

    @staticmethod
    def rot_and_shift(x, k):
        assert 0 < k < 4
        assert x.shape[-1] == x.shape[-2]
        assert x.shape[-3] == 4
        shape_bac = x.shape
        x = x.reshape(-1, *x.shape[-3:])
        x = torch.rot90(x, k, (2, 3))
        x = torch.cat((x[:, -k:, :, :], x[:, :-k, :, :]), dim=1)
        x = x.view(*shape_bac)
        return x

    def forward(self, x):
        p = self.padding
        B = x.shape[0]
        ch = x.shape[2]
        w = x.shape[-1]
        
        if len(x.shape) == 6:
            assert x.shape == (B, 6, ch, 4, w, w)
            z = torch.zeros((*x.shape[:-2], w + 2 * p, w + 2 * p), dtype=x.dtype, device=x.device)
            x = [x[:, i] for i in range(6)]
            z[:, 0] = CubePad.square_pad(x[0], x[4], x[3], x[5], x[1], pad=p)
            z[:, 1] = CubePad.square_pad(x[1], CubePad.rot_and_shift(x[4], 1), x[0], CubePad.rot_and_shift(x[5], 3), x[2], pad=p)
            z[:, 2] = CubePad.square_pad(x[2], CubePad.rot_and_shift(x[4], 2), x[1], CubePad.rot_and_shift(x[5], 2), x[3], pad=p)
            z[:, 3] = CubePad.square_pad(x[3], CubePad.rot_and_shift(x[4], 3), x[2], CubePad.rot_and_shift(x[5], 1), x[0], pad=p)
            z[:, 4] = CubePad.square_pad(x[4], CubePad.rot_and_shift(x[2], 2), CubePad.rot_and_shift(x[3], 1), x[0], CubePad.rot_and_shift(x[1], 3), pad=p)
            z[:, 5] = CubePad.square_pad(x[5], x[0], CubePad.rot_and_shift(x[3], 3), CubePad.rot_and_shift(x[2], 2), CubePad.rot_and_shift(x[1], 1), pad=p)
            return z
        elif len(x.shape) == 5:
            assert x.shape == (B, 6, ch, w, w)
            # TODO: update this part
            x = F.pad(x, pad=(p, p, p, p))
            x[:, 0, :, :p, p:-p]  += x[:, 3, :, -2 * p: -p, p:-p]
            x[:, 0, :, -p:, p:-p] += x[:, 1, :, p: 2 * p, p:-p]
            x[:, 0, :, p:-p, :p]  += x[:, 4, :, p:-p, -2 * p: -p]
            x[:, 0, :, p:-p, -p:] += x[:, 5, :, p:-p, p: 2 * p]
            x[:, 1, :, :p, p:-p]  += x[:, 0, :, -2 * p: -p, p:-p]
            x[:, 1, :, -p:, p:-p] += x[:, 2, :, p: 2 * p, p:-p]
            x[:, 1, :, p:-p, :p]  += torch.flip(torch.transpose(x[:, 4, :, -2 * p: -p, p:-p], 2, 3), [2])
            x[:, 1, :, p:-p, -p:] += torch.flip(torch.transpose(x[:, 5, :, -2 * p: -p, p:-p], 2, 3), [3])
            x[:, 2, :, :p, p:-p]  += x[:, 1, :, -2 * p: -p, p:-p]
            x[:, 2, :, -p:, p:-p] += x[:, 3, :, p: 2 * p, p:-p]
            x[:, 2, :, p:-p, :p]  += torch.flip(x[:, 4, :, p:-p, p: 2 * p], [2, 3])
            x[:, 2, :, p:-p, -p:] += torch.flip(x[:, 5, :, p:-p, -2 * p: -p], [2, 3])
            x[:, 3, :, :p, p:-p]  += x[:, 2, :, -2 * p: -p, p:-p]
            x[:, 3, :, -p:, p:-p] += x[:, 0, :, p: 2 * p, p:-p]
            x[:, 3, :, p:-p, :p]  += torch.flip(torch.transpose(x[:, 4, :, p: 2 * p, p:-p], 2, 3), [3])
            x[:, 3, :, p:-p, -p:] += torch.flip(torch.transpose(x[:, 5, :, p: 2 * p, p:-p], 2, 3), [2])
            x[:, 4, :, :p, p:-p]  += torch.flip(torch.transpose(x[:, 3, :, p:-p, p: 2 * p], 2, 3), [2])
            x[:, 4, :, -p:, p:-p] += torch.flip(torch.transpose(x[:, 1, :, p:-p, p: 2 * p], 2, 3), [3])
            x[:, 4, :, p:-p, :p]  += torch.flip(x[:, 2, :, p:-p, p: 2 * p], [2, 3])
            x[:, 4, :, p:-p, -p:] += x[:, 0, :, p:-p, p: 2 * p]
            x[:, 5, :, :p, p:-p]  += torch.flip(torch.transpose(x[:, 3, :, p:-p, -2 * p: -p], 2, 3), [3])
            x[:, 5, :, -p:, p:-p] += torch.flip(torch.transpose(x[:, 1, :, p:-p, -2 * p: -p], 2, 3), [2])
            x[:, 5, :, p:-p, :p]  += x[:, 0, :, p:-p, -2 * p: -p]
            x[:, 5, :, p:-p, -p:] += torch.flip(x[:, 2, :, p:-p, -2 * p: -p], [2, 3])
            assert x.shape == (B, 6, ch, w + 2 * p, w + 2 * p)
            return x
        else:
            assert False

    def __repr__(self):
        return 'CubePad(padding=%d)' % self.padding


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()
