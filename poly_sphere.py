import numpy as np
from utils import *


def refine_square(ul, ur, bl, br, w, project):
    if w == 1:
        out = np.array([[ul, ur], [bl, br]], dtype=np.float64)
        assert out.shape == (2, 2, 3)
        return out

    div = smallest_prime_factor(w)
    # NOTE: divisions that are closer to the center should be smaller but I've ignored that
    points = np.empty((div + 1, div + 1, 3), dtype=np.float64)
    for i in range(div + 1):
        for j in range(div + 1):
            points[i, j] = (i * j * br + i * (div - j) * bl + j * (div - i) * ur + (div - i) * (div - j) * ul) / (div ** 2)
            if project:
                points[i, j] = proj_unit(points[i, j])
    w_ = w // div
    out = np.empty((w + 1, w + 1, 3), dtype=np.float64)
    for i in range(div):
        for j in range(div):
            out[i * w_: (i + 1) * w_ + 1, j * w_: (j + 1) * w_ + 1] = refine_square(
                points[i, j], points[i, j + 1], points[i + 1, j], points[i + 1, j + 1], w_, project)
    return out


def refine_triangle(ul, ur, bo, w, project):
    if w == 1:
        out = np.array([[ul, ur], [bo, np.ones(3) * -1e9]], dtype=np.float64)
        assert out.shape == (2, 2, 3)
        return out

    div = largest_prime_factor(w)
    # NOTE: divisions that are closer to the center should be smaller but I've ignored that
    points = np.empty((div + 1, div + 1, 3), dtype=np.float64)
    for i in range(div + 1):
        for j in range(div + 1 - i):
            points[i, j] = (i * bo + (div - i - j) * ul + j * ur) / div
            if project:
                points[i, j] = proj_unit(points[i, j])

    w_ = w // div
    out = np.ones((w + 1, w + 1, 3), dtype=np.float64) * -1e9
    for i in range(div):
        for j in range(div - i):
            out[i * w_: (i + 1) * w_ + 1, j * w_: (j + 1) * w_ + 1] = np.maximum(
                out[i * w_: (i + 1) * w_ + 1, j * w_: (j + 1) * w_ + 1],
                refine_triangle(points[i, j], points[i, j + 1], points[i + 1, j], w_, project)
            )
    for i in range(div):
        for j in range(div - i - 1):
            out[i * w_: (i + 1) * w_ + 1, j * w_: (j + 1) * w_ + 1] = np.maximum(
                out[i * w_: (i + 1) * w_ + 1, j * w_: (j + 1) * w_ + 1],
                np.rot90(refine_triangle(points[i + 1, j + 1], points[i + 1, j], points[i, j + 1], w_, project), 2)
            )
    return out


def triangle_mask(w):
    mask = np.ones((w, w), dtype=np.uint8)
    x = np.arange(w)[None, :] * mask
    y = np.arange(w)[:, None] * mask
    mask *= (y < w - x)
    return mask


def get_centers_of_square(x):
    x = (x[:-1,:-1] + x[:-1,1:] + x[1:,:-1] + x[1:,1:]) / 4
    return x / np.linalg.norm(x, ord=2, axis=2, keepdims=True)


def get_centers_of_triangle(z):
    z = (z[:-1,:-1] + z[:-1,1:] + z[1:,:-1]) / 3
    z = z / np.linalg.norm(z, ord=2, axis=2, keepdims=True)
    w = z.shape[1]
    z = z * triangle_mask(w)[:, :, None]
    return z


def make_cube(face):
    w = face.shape[1]
    cube = np.empty((6, w * w, 3), dtype=float)
    face = face.reshape((w * w, 3))
    cube[0] = face
    cube[1] = rotate_3d(face, 1 * np.pi / 2, axis=0)
    cube[2] = rotate_3d(face, 2 * np.pi / 2, axis=0)
    cube[3] = rotate_3d(face, 3 * np.pi / 2, axis=0)
    cube[4] = rotate_3d(face, 1 * np.pi / 2, axis=1)
    cube[5] = rotate_3d(face, 3 * np.pi / 2, axis=1)
    cube = cube.reshape(6, w, w, 3)
    return cube


def make_icosa(face):
    w = face.shape[1]
    a = np.sqrt(9 * np.tan(np.pi / 5) ** 2 - 3) # triangle side length
    h = np.sqrt(3) / 2 * a
    phi = 2 * np.arctan(h / 3)

    w00 = face.reshape(-1, 3)
    w01 = rotate_3d(w00, np.pi, axis=2)
    w01 = rotate_3d(w01,  -phi, axis=0)
    w02 = rotate_3d(w01, +2 * np.pi / 3, axis=2)
    w03 = rotate_3d(w01, -2 * np.pi / 3, axis=2)
    w05 = rotate_3d(w02, np.pi, axis=2)
    w05 = rotate_3d(w05,  -phi, axis=0)
    w06 = rotate_3d(w03, np.pi, axis=2)
    w06 = rotate_3d(w06,  -phi, axis=0)
    w04 = rotate_3d(w05, +2 * np.pi / 3, axis=2)
    w07 = rotate_3d(w05, -2 * np.pi / 3, axis=2)
    w08 = rotate_3d(w06, -2 * np.pi / 3, axis=2)
    w09 = rotate_3d(w06, +2 * np.pi / 3, axis=2)

    w10 = rotate_3d(w00, -np.pi, axis=0)
    w11 = rotate_3d(w01, -np.pi, axis=0)
    w12 = rotate_3d(w02, -np.pi, axis=0)
    w13 = rotate_3d(w03, -np.pi, axis=0)
    w14 = rotate_3d(w04, -np.pi, axis=0)
    w15 = rotate_3d(w05, -np.pi, axis=0)
    w16 = rotate_3d(w06, -np.pi, axis=0)
    w17 = rotate_3d(w07, -np.pi, axis=0)
    w18 = rotate_3d(w08, -np.pi, axis=0)
    w19 = rotate_3d(w09, -np.pi, axis=0)

    all_w = [rotate_3d(x, -np.arctan(2 * h / 3), axis=0) for x in [w00, w01, w02, w03, w04, w05, w06, w07, w08, w09, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19]]
    icosa = np.stack(all_w, axis=0)
    icosa = icosa.reshape(20, w, w, 3)
    return icosa


def make_tetra(face):
    w = face.shape[1]
    a = 2 * np.sqrt(6) # triangle side length
    h = np.sqrt(3) / 2 * a
    phi = 2 * np.arctan(h / 3)

    w00 = face.reshape(-1, 3)
    w01 = rotate_3d(w00, np.pi, axis=2)
    w01 = rotate_3d(w01,  -phi, axis=0)
    w02 = rotate_3d(w01, +2 * np.pi / 3, axis=2)
    w03 = rotate_3d(w01, -2 * np.pi / 3, axis=2)

    all_w = [rotate_3d(x, -np.arctan(2 * h / 3), axis=0) for x in [w00, w01, w02, w03]]    
    tetra = np.stack(all_w, axis=0)
    tetra = tetra.reshape(4, w, w, 3)
    return tetra


def make_octa(face):
    w = face.shape[1]
    a = 2 * np.sqrt(6) # triangle side length
    h = np.sqrt(3) / 2 * a
    phi = 2 * np.arctan(h / 3)

    w00 = face.reshape(-1, 3)
    w01 = rotate_3d(w00, np.pi / 2, axis=2)
    w02 = rotate_3d(w01, np.pi / 2, axis=2)
    w03 = rotate_3d(w02, np.pi / 2, axis=2)
    w04 = rotate_3d(w00, np.pi, axis=0)
    w05 = rotate_3d(w01, np.pi, axis=0)
    w06 = rotate_3d(w02, np.pi, axis=0)
    w07 = rotate_3d(w03, np.pi, axis=0)

    octa = np.stack([w00, w01, w02, w03, w04, w05, w06, w07], axis=0)
    octa = octa.reshape(8, w, w, 3)
    return octa


def get_sampling_grid(polyhedron, w, center='True'):
    if polyhedron == 'tetra':
        a = 2 * np.sqrt(6) # triangle side length
        ul = proj_unit(np.array([-a / 2, a / (2 * np.sqrt(3)), 1], dtype=float))
        ur = proj_unit(np.array([+a / 2, a / (2 * np.sqrt(3)), 1], dtype=float))
        bo = proj_unit(np.array([0, -a / np.sqrt(3), 1], dtype=float))

        x = refine_triangle(ul, ur, bo, w, project=True)
        if center:
            x = get_centers_of_triangle(x)
        x = make_tetra(x)
    elif polyhedron == 'cube':
        ul = proj_unit(np.array([-1, +1, +1], dtype=float))
        ur = proj_unit(np.array([+1, +1, +1], dtype=float))
        bl = proj_unit(np.array([-1, -1, +1], dtype=float))
        br = proj_unit(np.array([+1, -1, +1], dtype=float))

        x = refine_square(ul, ur, bl, br, w, project=True)
        if center:
            x = get_centers_of_square(x)
        x = make_cube(x)
    elif polyhedron == 'octa':
        ul = proj_unit(np.array([0, -1, 0], dtype=float))
        ur = proj_unit(np.array([+1, 0, 0], dtype=float))
        bo = proj_unit(np.array([0, 0, -1], dtype=float))

        x = refine_triangle(ul, ur, bo, w, project=True)
        if center:
            x = get_centers_of_triangle(x)
        x = make_octa(x)
    elif polyhedron == 'icosa':
        a = np.sqrt(9 * np.tan(np.pi / 5) ** 2 - 3) # triangle side length
        ul = proj_unit(np.array([-a / 2, a / (2 * np.sqrt(3)), 1], dtype=float))
        ur = proj_unit(np.array([+a / 2, a / (2 * np.sqrt(3)), 1], dtype=float))
        bo = proj_unit(np.array([0, -a / np.sqrt(3), 1], dtype=float))

        x = refine_triangle(ul, ur, bo, w, project=True)
        if center:
            x = get_centers_of_triangle(x)
        x = make_icosa(x)
    else:
        assert False

    return x


def equirectangular_to_polysphere(X, spherical_sampling_points, interpolation):
    # assumption: `X` has shape [B, H, W, C]
    H = X.shape[1]
    W = X.shape[2]

    # extend image so that `bilinear_interpolate` can work on edge pixels
    X = np.concatenate([X, X[:, -1:, :]], axis=1)
    X = np.concatenate([X, X[:, :, :2]], axis=2)

    s = spherical_sampling_points
    y = s[:, 0] / np.pi * (H - 1)
    x = np.where(s[:, 1] < 0, s[:, 1] + 2 * np.pi, s[:, 1]) / (2 * np.pi) * W
    
    if interpolation == 'bilinear':
        Y = bilinear_interpolate(X.astype(np.float64), x, y).astype(X.dtype)
    elif interpolation == 'nearest':
        Y = nearest_interpolate(X, x, y)
    else:
        assert False, 'ERROR: Unknown interpolation method `%s`.' % interpolation

    return Y
