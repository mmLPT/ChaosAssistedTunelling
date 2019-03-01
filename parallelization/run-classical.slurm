#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:02:00
#SBATCH --job-name="SPS CLASS"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=121
name=10-moyen
wdir=data/output/$name-classical-${SLURM_JOB_ID}
inputfile=data/input/$name

srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --out-dir $wdir --command "python3 classical.py compute $wdir/ $inputfile $nruns %path%"
python3 classical.py plot $wdir/ $nruns



