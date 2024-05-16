#!/usr/bin/bash -l

#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=ccm
#SBATCH --output=output_slurm/%j_output
#SBATCH --error=output_slurm/%j_error

conda deactivate
source /mnt/home/rohana/python_envs/nanoGPT/bin/activate

python transform.py