#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:2
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=1-30%1

./run_happi20.sh $@
