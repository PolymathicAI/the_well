#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=32
#SBATCH -t 01:00:00
#SBATCH --output=training-%j.log
#SBATCH -J "well_benchmark"

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

module load python cuda cudnn
source ~/well_venv/bin/activate
srun python train.py experiment=fno server=rusty trainer=debug