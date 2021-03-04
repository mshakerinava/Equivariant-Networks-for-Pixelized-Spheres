#!/bin/bash

source /home/mehrans/venv_py36/bin/activate
cd $SLURM_TMPDIR
git clone https://github.com/mshakerinava/Equivariant-Networks-for-Pixelized-Spheres.git
cd Equivariant-Networks-for-Pixelized-Spheres
cp -rf /home/mehrans/scratch/smnist_poly .
git config user.email mehranshakerinava@gmail.com && git config user.name "Mehran Shakerinava"
mkdir -p /home/mehrans/scratch/checkpoints/
python3 polynet.py $@ --checkpoint_path /home/mehrans/scratch/checkpoints/
if [ $? -eq 0 ]; then
    git pull && git add logs && git commit -m "commit logs" && git push
else
    echo "ERROR: not committing"
fi
mkdir -p /home/mehrans/scratch/saved_models/
cp *.tar /home/mehrans/scratch/saved_models/
