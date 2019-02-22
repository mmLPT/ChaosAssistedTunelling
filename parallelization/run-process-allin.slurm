#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:10:00
#SBATCH --job-name="ALLIN PROC"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nh=200
name=allin1-125233

srun chdb --in-type "1 $nh" --command "python3 all-in.py process data/output/$name/ %path%"

