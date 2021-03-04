#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load httpproxy
./run_2d3ds.sh $@
