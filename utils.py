import json
import hashlib
import numpy as np


def hash_args(args_dict, no_hash):
    args_dict = {x: args_dict[x] for x in args_dict if x not in no_hash}
    args_str = json.dumps(args_dict, sort_keys=True, indent=4)
    args_hash = hashlib.md5(str.encode(args_str)).hexdigest()[:8]
    return args_hash


def largest_prime_factor(x):
    assert x >= 1
    if x == 1:
        return 1
    i = 2
    while True:
        while x % i == 0:
            x //= i
        if x == 1:
            break
        i += 1
    return i


def smallest_prime_factor(x):
    assert x >= 1
    if x == 1:
        return 1
    for i in range(2, x + 1):
        if x % i == 0:
            break
    return i


def rotate_2d(u, theta):
    n = u.shape[0]
    assert u.shape == (n, 2)
    v = u[:, 0] + u[:, 1] * 1j
    r = np.cos(theta) + np.sin(theta) * 1j
    v *= r
    return np.stack([v.real, v.imag], axis=1)


def rotate_3d(u, theta, axis):
    n = u.shape[0]
    assert u.shape == (n, 3)
    assert axis in [0, 1, 2]
    u = u.copy()
    idx = [i for i in range(3) if i != axis]
    u[:, idx] = rotate_2d(u[:, idx], theta)
    return u


def proj_unit(x):
    return x / np.linalg.norm(x, ord=2)


def cartesian_to_spherical(u):
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r_ = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(r_, z)
    phi = np.arctan2(y, x)
    return np.stack([r, theta, phi], axis=1)


def spherical_to_cartesian(s):
    r, theta, phi = s[:, 0], s[:, 1], s[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


# adapted from `https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python`
def bilinear_interpolate(im, x, y):
    # assumption: `im` has shape [B, H, W, C]
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    assert (0 <= x0).all() and (x0 <= im.shape[2] - 1).all()
    assert (0 <= x1).all() and (x1 <= im.shape[2] - 1).all()
    assert (0 <= y0).all() and (y0 <= im.shape[1] - 1).all()
    assert (0 <= y1).all() and (y1 <= im.shape[1] - 1).all()

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    wa = ((x1 - x) * (y1 - y))[None, :, None]
    wb = ((x1 - x) * (y - y0))[None, :, None]
    wc = ((x - x0) * (y1 - y))[None, :, None]
    wd = ((x - x0) * (y - y0))[None, :, None]

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


# adapted from `https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python`
def nearest_interpolate(im, x, y):
    # assumption: `im` has shape [B, H, W, ...]
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.round(x).astype(int)
    y0 = np.round(y).astype(int)

    assert (0 <= x0).all() and (x0 <= im.shape[2] - 1).all()
    assert (0 <= y0).all() and (y0 <= im.shape[1] - 1).all()

    return im[:, y0, x0]
