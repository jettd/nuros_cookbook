#!/bin/bash
#SBATCH --job-name=food101_efficientnet_1
#SBATCH --account=jdurante2
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail; mkdir -p logs
export PYENV_VERSION=anaconda3-2024.10-1
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "Node: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi

python /home/jdurante2/guide/nuros_cookbook/examples/food101_example.py
