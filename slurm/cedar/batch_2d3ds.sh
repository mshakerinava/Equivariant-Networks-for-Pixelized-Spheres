#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

./run_2d3ds.sh $@
