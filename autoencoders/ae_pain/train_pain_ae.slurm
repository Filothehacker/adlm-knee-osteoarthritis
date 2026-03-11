#!/bin/bash

#SBATCH --job-name=knee_pain_ae_train
#SBATCH --output=train_pain_ae_%j.out
#SBATCH --error=train_pain_ae_%j.err
#SBATCH --qos=master-queuesave
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G               

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

cd /vol/miltank/users/$USER/git/adlm-knee-osteoarthritis

echo "STARTING PAIN-SUPERVISED AE TRAINING"
echo "Running ae_pain.train..."

python autoencoders/ae_pain/train.py

echo "Training DONE"