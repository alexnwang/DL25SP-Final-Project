#!/bin/bash
#SBATCH --partition=a100_short,radiology,a100_long,gpu4_medium,gpu4_long,gpu8_medium,gpu8_long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=hcc
#SBATCH --output=/gpfs/scratch/wz1492/DL25SP-Final-Project/logs/hparam_search_%A_%a.out
#SBATCH --array=1-20%5

# load modules and environment
module load gcc/8.1.0
source ~/.bashrc
conda activate dino_wm

# set a reproducible seed based on array task ID
seed=$((SLURM_ARRAY_TASK_ID + 1000))

# define ViT size categories
sizes=(tiny small base large)
# randomly choose one size category
size=${sizes[$((RANDOM % ${#sizes[@]}))]}

# assign hyperparameters based on chosen size
case $size in
    super_tiny)
        depth=4
        heads=4
        dim_head=32
        mlp_dim=512
        ;;
    tiny)
        depth=12
        heads=3
        dim_head=64
        mlp_dim=768
        ;;
    small)
        depth=12
        heads=6
        dim_head=64
        mlp_dim=1536
        ;;
    base)
        depth=12
        heads=12
        dim_head=64
        mlp_dim=3072
        ;;
esac

# randomly pick dropout rate
dropouts=(0.0 0.05 0.1)
dropout=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

# randomly pick scheduled sampling probability
sched_probs=(0.3 0.5 0.7 1.0)
sched_prob=${sched_probs[$((RANDOM % ${#sched_probs[@]}))]}

# define learning rate options and random selection
lrs=(1e-5 5e-5 1e-4 5e-4 1e-3)
lr=${lrs[$((RANDOM % ${#lrs[@]}))]}

# construct run name from hyperparameters
run_name="size-${size}_d${depth}_h${heads}_hd${dim_head}_mlp${mlp_dim}_do${dropout}_sched${sched_prob}_lr${lr}_seed${seed}"

# set batch size based on size
if [ "$size" = "base" ]; then
  batch_size=16
else
  batch_size=16
fi

# display chosen configuration
echo "Task $SLURM_ARRAY_TASK_ID | Size: $size | depth=$depth | heads=$heads | dim_head=$dim_head | mlp_dim=$mlp_dim | dropout=$dropout | sched_sample_prob=$sched_prob | lr=$lr | vicreg=$vicreg_name | seed=$seed"

# run training
python train_jepa.py \
  --data-dir /gpfs/scratch/wz1492/data/train \
  --epochs 50 \
  --batch-size $batch_size \
  --num-hist 16 \
  --lr $lr \
  --predictor-depth $depth \
  --predictor-heads $heads \
  --predictor-dim-head $dim_head \
  --predictor-mlp-dim $mlp_dim \
  --predictor-dropout $dropout \
  --predictor-emb-dropout $dropout \
  --sched-sample-prob $sched_prob \
  --run-name $run_name \
  --save-path /gpfs/scratch/wz1492/DL25SP-Final-Project/checkpoints/${run_name}.pth 