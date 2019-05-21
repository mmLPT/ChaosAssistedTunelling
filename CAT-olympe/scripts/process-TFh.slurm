#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:10:00
#SBATCH --job-name="ALLIN PROC"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nh=200
name=e010-g050-allin-148161

srun chdb --in-type "1 $nh" --command "python3 TFh.py process /tmpdir/p0110mm/$name/ %path%"

