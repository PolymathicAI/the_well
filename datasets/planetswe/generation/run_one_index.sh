#!/bin/bash -l
#SBATCH --time=3:00:00
#SBATCH -p polymathic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -J run_all_swe
#SBATCH -C icelake

 # 5, 14, 15, 19
module load modules/2.3-beta1  openmpi/4.0.7 dedalus/3.2302-dev-py3.10.13
srun python gen_SWE_from_ic_file.py --index $1
