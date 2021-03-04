#!/bin/bash

source /home/mehrans/venv_py36/bin/activate
cd $SLURM_TMPDIR
git clone https://github.com/mshakerinava/Equivariant-Networks-for-Pixelized-Spheres.git
cd Equivariant-Networks-for-Pixelized-Spheres

mkdir -p happi20_poly
date
cp /home/mehrans/scratch/happi20_poly/cube_48.tar happi20_poly/
cd happi20_poly
date
tar -xf cube_48.tar
date
cd ..

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
