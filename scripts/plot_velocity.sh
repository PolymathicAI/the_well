#!/usr/bin/bash -l

#SBATCH --partition=polymathic
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --output=plot_well_data_%j.out

module load python
source ~/well_venv/bin/activate

python -u plot_velocity.py $@
