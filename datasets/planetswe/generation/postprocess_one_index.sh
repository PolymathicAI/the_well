#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH -p polymathic
#SBATCH --array=0-39 -N1 -c1 --mem=10000 -o arrayExample.%j.%A.%a.%N.out -e arrayExample.%j.%A.%a.%N.err
# # SBATCH --nodes=1
# # SBATCH --ntasks=96
# # SBATCH --ntasks-per-core=1
# # SBATCH -J run_all_swe
#SBATCH -C genoa

 # 5, 14, 15, 19
module load modules/2.2-20230808  openmpi/4.0.7 dedalus/3.2302-dev-py3.10.10

# export OMP_NUM_THREADS=1
# export NUMEXPR_MAX_THREADS=1


python interpolate_swe_data.py --index $SLURM_ARRAY_TASK_ID
