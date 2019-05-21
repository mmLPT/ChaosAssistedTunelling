#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --time=00:05:00
#SBATCH --job-name="ALLIN COMP"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nparams=20
name=explore2-explore-eg-141926

srun chdb --in-type "1 $nparams" --command "python3 explore-eg.py process /tmpdir/p0110mm/$name/ %path%"
