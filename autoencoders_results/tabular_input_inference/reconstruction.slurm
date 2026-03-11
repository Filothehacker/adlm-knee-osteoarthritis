#!/bin/bash

#SBATCH --job-name=tabular_input_ae_reconstruction
#SBATCH --output=tabular_input_ae_reconstruction_%j.out
#SBATCH --error=tabular_input_ae_reconstruction_%j.err
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

cd /vol/miltank/users/$USER/Documents/adlm-knee-osteoarthritis

echo "STARTING TABULAR-INPUT AE RECONSTRUCTION"
echo "Running tabular_input_inference/reconstruction.py..."

python autoencoders_results/tabular_input_inference/reconstruction.py

echo "Reconstruction DONE"