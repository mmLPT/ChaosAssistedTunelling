#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:05:00
#SBATCH --job-name="GPHY"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=36
name=3levels
wdir=/tmpdir/p0110mm/$name-gphy-${SLURM_JOB_ID}
inputfile=input/$name

srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --out-dir $wdir --command "python3 getphysics.py compute $wdir/ $inputfile $nruns %path%"



