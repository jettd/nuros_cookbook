#!/bin/bash
#SBATCH --job-name=food101_efficientnet
#SBATCH --account=USERNAME
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

mkdir -p logs #make sure log location exists
export PYENV_VERSION=anaconda3-2024.10-1 #necessary for Nuros
export TF_FORCE_GPU_ALLOW_GROWTH=true #necessary to avoid GPU memory allocation issues on Nuros

# Dont forget to change --account=USERNAME (at the top) and the path in the line below
python /home/USERNAME/path/to/file.py
#example could be: python /home/jdoe2/scripts/train_model.py

