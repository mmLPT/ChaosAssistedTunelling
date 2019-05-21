#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=24:00:00
#SBATCH --job-name="ALLIN COMP"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=10000
ne=2
ngamma=10
name=explore2
wdir=/tmpdir/p0110mm/$name-explore-eg-${SLURM_JOB_ID}
inputfile=input/$name

python3 explore-eg.py initialize $wdir/ $ne $ngamma
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 explore-eg.py compute $wdir/ $inputfile $nruns $ne $ngamma %path%"
