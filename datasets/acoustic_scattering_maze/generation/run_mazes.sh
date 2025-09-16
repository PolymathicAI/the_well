#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH -p cmbas
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -J acou_maze
#SBATCH -o acou_maze
#SBATCH -C icelake

source ~/venvs/clawpack/bin/activate

srun python generate_acoustics_data.py --switch_to_maze
