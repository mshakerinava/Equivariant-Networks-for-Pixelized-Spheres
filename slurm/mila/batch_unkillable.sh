#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

./run_polynet.sh $@
