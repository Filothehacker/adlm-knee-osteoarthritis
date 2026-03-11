#!/bin/bash

#SBATCH --job-name=knee_tabular_input_ae_train
#SBATCH --output=train_tabular_input_ae_%j.out
#SBATCH --error=train_tabular_input_ae_%j.err
#SBATCH --qos=master-queuesave
#SBATCH --time=0-22:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G             
#SBATCH --exclude=helios  

source /etc/profile.d/modules.sh
module load python/uv

uv sync

source .venv/bin/activate

echo "Python interpreter:"
which python

echo "Python version:"
python --version

echo "GPU status:"
nvidia-smi
echo


echo "STARTING TABULAR-INPUT MASKED AE TRAINING"
echo "Running ae_tabular_input_masked/train.py..."

python autoencoders/ae_tabular_input_masked/train.py

echo "Training DONE"