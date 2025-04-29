#!/bin/bash
#SBATCH -p a100_short,radiology,a100_long,gpu4_medium,gpu4_long,gpu8_medium,gpu8_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --job-name=hcc
#SBATCH --output=/gpfs/scratch/wz1492/DL25SP-Final-Project/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate dino_wm

python train_jepa.py \
    --data-dir /gpfs/scratch/wz1492/data/train \
    --output-dir runs/test/ \
    --sched-sample-prob 0.5 \
    --epochs 12 \
    --lr 1e-4 \
    --predictor-pool attn \
    --batch-size 32 \



python main.py \
    --predictor-pool attn \

