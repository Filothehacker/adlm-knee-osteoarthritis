#!/bin/bash

#SBATCH --job-name=knee_ae_train
#SBATCH --output=train_ae%j.out
#SBATCH --error=train_ae%j.err
#SBATCH --time=1-00:00:00              # 1 day, adjust as needed
#SBATCH --gres=gpu:1                   # 1 GPU
#SBATCH --cpus-per-task=8              # dataloader workers, etc.
#SBATCH --mem=32G                      # increase for 224x224xD volumes
#SBATCH --nice=100                     # optional priority smoothing

# Load python uv module
ml python/uv

uv sync

# Activate the project environment (.venv created by uv)
source .venv/bin/activate

echo "Python interpreter:"
which python

echo "Python version:"
python --version

echo "GPU status:"
nvidia-smi
echo


# ----------------------------
# RUN TRAINING
# ----------------------------

echo "STARTING TRAINING JOB"
echo "Running train.py..."
python autoencoders/ae_filippo/train.py

echo "Training DONE"