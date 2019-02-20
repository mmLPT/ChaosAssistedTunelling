#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=1000
name=s-27.84-nu-80.0-Ndbeta-8.0-beta0=0.25
wdir=data/output/$name-${SLURM_JOB_ID}
inputfile=data/input/$name

srun chdb --in-type "0 $nruns" --report "$wdir/report.txt" --out-dir $wdir --command "python3 freepropaveraged.py compute $wdir/ $inputfile $nruns %path%"

