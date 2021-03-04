import argparse
from poly_sphere import *
from autoequiv import *


def symmetry_perm(poly, model_type, around_z):
    if model_type == 1:
        x = get_sampling_grid(poly, 1, center=True)
    elif model_type == 2:    
        x = get_sampling_grid(poly, 1, center=False)
        x = x.reshape((-1, 4, 3))
        if poly != 'cube':
            x = x[:, :3]
        # move points slightly toward center of face so that all points become distinct
        x = 0.99 * x + 0.01 * np.mean(x, axis=1, keepdims=True)
    else:
        assert False

    x = x.reshape((-1, 3))
    n = x.shape[0]

    if around_z:
        if poly == 'tetra':
            y = rotate_3d(x, 2 * np.pi / 3, axis=2)
        elif poly == 'cube' or poly == 'octa':
            y = rotate_3d(x, 2 * np.pi / 4, axis=2)
        elif poly == 'icosa':
            y = rotate_3d(x, 2 * np.pi / 5, axis=2)
        else:
            assert False
    else:
        if poly == 'tetra':
            y = rotate_3d(x, np.arccos(1 / 3) - np.pi, axis=0)
            y = rotate_3d(y, np.pi, axis=2)
        elif poly == 'cube' or poly == 'octa':
            y = rotate_3d(x, np.pi / 2, axis=0)
        elif poly == 'icosa':
            a = np.sqrt(9 * np.tan(np.pi / 5) ** 2 - 3) # triangle side length
            ul = proj_unit(np.array([-a / 2, a / (2 * np.sqrt(3)), 1], dtype=float))
            ur = proj_unit(np.array([+a / 2, a / (2 * np.sqrt(3)), 1], dtype=float))
            l = np.linalg.norm(ul - ur)
            y = rotate_3d(x, -2 * np.arcsin(l / 2), axis=0)
            y = rotate_3d(y, np.pi / 5, axis=2)

    perm = [None] * n
    for i in range(n):
        for j in range(n):
            if np.linalg.norm(x[i] - y[j], ord=2) < 1e-6:
                perm[i] = j
                break

    assert is_permutation(perm, n)
    return perm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--polyhedron', type=str, choices=['icosa', 'cube', 'octa', 'tetra'], required=True)
    parser.add_argument('--model_type', type=int, choices=[1, 2], required=True)
    parser.add_argument('--around_z', action='store_true')
    args = parser.parse_args()

    print(symmetry_perm(args.polyhedron, args.model_type, args.around_z))
