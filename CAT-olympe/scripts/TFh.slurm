#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=2:00:00
#SBATCH --job-name="ALLIN COMP"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=20000
nh=200
name=e020-g050
wdir=/tmpdir/p0110mm/$name-TFh-${SLURM_JOB_ID}
inputfile=input/$name

python3 TFh.py initialize $wdir/ $nh
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 TFh.py compute $wdir/ $inputfile $nruns $nh %path%"

