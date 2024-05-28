#!/bin/bash -l

#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=32
#SBATCH -t 10:00:00
#SBATCH --output=training-%j.log
#SBATCH -J "well_benchmark"

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}

module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
# source ~/well_venv/bin/activate
source /mnt/home/rohana/python_envs/nanoGPT/bin/activate

# Launch the training script
srun python train.py experiment=fno server=rusty optimizer.lr=1e-2