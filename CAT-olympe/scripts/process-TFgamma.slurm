#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:10:00
#SBATCH --job-name="ALLIN PROC"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

ngamma=100
name=observable2-allin-157777

srun chdb --in-type "1 $ngamma" --command "python3 TFgamma.py process /tmpdir/p0110mm/$name/ %path%"

