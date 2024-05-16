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
mkdir -p /tmp/the_well/2D/
cp -r /mnt/home/polymathic/ceph/the_well/2D/active_matter /tmp/the_well/2D/
srun python train.py experiment=debug data.well_base_path=/tmp/the_well/2D/ server=rusty