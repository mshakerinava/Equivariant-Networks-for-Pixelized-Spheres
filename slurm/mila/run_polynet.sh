#!/bin/bash

module --quiet load miniconda/3
conda activate conda_env
cd $SLURM_TMPDIR
git clone https://github.com/mshakerinava/Equivariant-Networks-for-Pixelized-Spheres.git
cd Equivariant-Networks-for-Pixelized-Spheres
cp -rf ~/smnist_poly .
cp -rf ~/2d3ds_poly .
git config user.email mehranshakerinava@gmail.com && git config user.name "Mehran Shakerinava"
export PYTHONUNBUFFERED=1
mkdir -p /miniscratch/mehran.shakerinava/checkpoints/
python3 polynet.py $@ --checkpoint_path /miniscratch/mehran.shakerinava/checkpoints/
if [ $? -eq 0 ]; then
    git pull && git add logs && git commit -m "commit logs" && git push
else
    echo "ERROR: not committing"
fi
mkdir -p /miniscratch/mehran.shakerinava/saved_models/
cp *.tar /miniscratch/mehran.shakerinava/saved_models/
