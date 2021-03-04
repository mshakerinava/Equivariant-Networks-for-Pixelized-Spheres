# Spherical MNIST

`gendata.py` was taken from https://github.com/jonas-koehler/s2cnn/tree/master/examples/mnist. I changed `import lie_learn.spaces.S2 as S2` to `import S2` and also set a seed for the RNG before calling `main` so that the same dataset is generated on each run.

`S2.py` was taken from https://github.com/AMLab-Amsterdam/lie_learn/tree/master/lie_learn/spaces.

