#!/bin/bash

python3 proj_smnist.py --poly cube --input smnist/s2_mnist_rr.gz --output smnist_poly/cube_24_rr --width 24
# python3 proj_smnist.py --poly cube --input smnist/s2_mnist_nr.gz --output smnist_poly/cube_24_nr --width 24
# python3 proj_smnist.py --poly cube --input smnist/s2_mnist_rn.gz --output smnist_poly/cube_24_rn --width 24
# python3 proj_smnist.py --poly cube --input smnist/s2_mnist_nn.gz --output smnist_poly/cube_24_nn --width 24

python3 proj_smnist.py --poly tetra --input smnist/s2_mnist_rr.gz --output smnist_poly/tetra_41_rr --width 41
# python3 proj_smnist.py --poly tetra --input smnist/s2_mnist_nr.gz --output smnist_poly/tetra_41_nr --width 41
# python3 proj_smnist.py --poly tetra --input smnist/s2_mnist_rn.gz --output smnist_poly/tetra_41_rn --width 41
# python3 proj_smnist.py --poly tetra --input smnist/s2_mnist_nn.gz --output smnist_poly/tetra_41_nn --width 41

python3 proj_smnist.py --poly octa --input smnist/s2_mnist_rr.gz --output smnist_poly/octa_25_rr --width 25
# python3 proj_smnist.py --poly octa --input smnist/s2_mnist_nr.gz --output smnist_poly/octa_25_nr --width 25
# python3 proj_smnist.py --poly octa --input smnist/s2_mnist_rn.gz --output smnist_poly/octa_25_rn --width 25
# python3 proj_smnist.py --poly octa --input smnist/s2_mnist_nn.gz --output smnist_poly/octa_25_nn --width 25

python3 proj_smnist.py --poly icosa --input smnist/s2_mnist_rr.gz --output smnist_poly/icosa_17_rr --width 17
# python3 proj_smnist.py --poly icosa --input smnist/s2_mnist_nr.gz --output smnist_poly/icosa_17_nr --width 17
# python3 proj_smnist.py --poly icosa --input smnist/s2_mnist_rn.gz --output smnist_poly/icosa_17_rn --width 17
# python3 proj_smnist.py --poly icosa --input smnist/s2_mnist_nn.gz --output smnist_poly/icosa_17_nn --width 17
