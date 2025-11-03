#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J acou_inccont
#SBATCH -o acou_inccont
#SBATCH -C icelake

source ~/venvs/clawpack/bin/activate

srun python generate_acoustics_data.py --inclusions
