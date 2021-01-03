#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=msc

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --array=0-5

#SBATCH --job-name="rand-pl"

export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p $TMPDIR

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml

source /scratch-ssd/oatml/miniconda3/bin/activate ml4

SEEDS=(42 24 1008 7 2020 96)

START=$(date +%s.%N)
srun python train.py \
    --alpha 0.4 \
    --b 40 \
    --iters 6 \
    --reps 1 \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "run time: $DIFF secs"

# RUN THIS SCRIPT USING: sbatch <this script's name>.sh
