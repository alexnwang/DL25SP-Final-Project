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
#SBATCH --array=1-10

# load modules and environment
module load gcc/8.1.0
source ~/.bashrc
conda activate dino_wm

# set a reproducible seed based on array task ID
seed=$((SLURM_ARRAY_TASK_ID + 1000))

# define ViT size categories
sizes=(tiny super_tiny)
# randomly choose one size category
size=${sizes[$((RANDOM % ${#sizes[@]}))]}

# assign hyperparameters based on chosen size
case $size in
    super_tiny)
        depth=2
        heads=4
        dim_head=32
        mlp_dim=512
        ;;
    tiny)
        depth=4
        heads=4
        dim_head=32
        mlp_dim=512
        ;;
esac

# randomly pick dropout rate
dropouts=(0.0 0.05)
dropout=${dropouts[$((RANDOM % ${#dropouts[@]}))]}

# randomly pick scheduled sampling probability
sched_probs=(0.3 0.5 0.7)
sched_prob=${sched_probs[$((RANDOM % ${#sched_probs[@]}))]}

# define learning rate options and random selection
# Define min and max learning rates
min_lr=1e-5
max_lr=1e-4

# Generate a random float between 0 and 1
rand=$(awk -v seed=$RANDOM 'BEGIN { srand(seed); print rand() }')

# Scale it to the desired range
lr=$(awk -v r=$rand -v min=$min_lr -v max=$max_lr 'BEGIN { print min + r * (max - min) }')


# construct run name from hyperparameters and task ID
run_name="task${SLURM_ARRAY_TASK_ID}_size-${size}_d${depth}_h${heads}_hd${dim_head}_mlp${mlp_dim}_do${dropout}_sched${sched_prob}_lr${lr}_seed${seed}"

# set batch size based on size
if [ "$size" = "base" ]; then
  batch_size=32
else
  batch_size=32
fi

# create unique directory for this run
output_dir="/gpfs/scratch/wz1492/DL25SP-Final-Project/runs/${SLURM_JOB_ID}/${run_name}"
mkdir -p $output_dir

# display chosen configuration
echo "Task $SLURM_ARRAY_TASK_ID | Size: $size | depth=$depth | heads=$heads | dim_head=$dim_head | mlp_dim=$mlp_dim | dropout=$dropout | sched_sample_prob=$sched_prob | lr=$lr | vicreg=$vicreg_name | seed=$seed | Output Dir: $output_dir"

# run training
python train_jepa.py \
  --data-dir /gpfs/scratch/wz1492/data/train \
  --epochs 15 \
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
  --output-dir ${output_dir} 