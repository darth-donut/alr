#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=msc

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

#SBATCH --job-name="pmix"

export TMPDIR=/scratch-ssd/${USER}/tmp
mkdir -p $TMPDIR

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/scripts/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml

source /scratch-ssd/oatml/miniconda3/bin/activate ml3

START=$(date +%s.%N)
srun python train.py \
    --acq "random" \
    --alpha 0.4 \
    --b 400 \
    --reps 1 \
    --iters 6

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "run time: $DIFF secs"

# RUN THIS SCRIPT USING: sbatch <this script's name>.sh
