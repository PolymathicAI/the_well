#!/usr/bin/bash -l

#SBATCH --partition=polymathic
#SBATCH -C genoa
#SBATCH --time=20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=96
#SBATCH --output=upload_well_data_%j.out


module load python
module load hdf5
source ~/venvs/well_venv/bin/activate

set -x

huggingface-cli login --token $HF_TOKEN --add-to-git-credential
srun python -u upload.py $@
