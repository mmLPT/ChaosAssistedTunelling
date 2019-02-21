#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 3
#SBATCH -n 108
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=4500
nh=900
name=allin1
#wdir=data/output/$name-${SLURM_JOB_ID}
inputfile=data/input/$name
wdir=data/output/allin1-124813

#python3 all-in.py initialize $wdir/ $nh
#srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 all-in.py compute $wdir/ $inputfile $nruns $nh %path%"
srun chdb --in-type "1 $nh" --command "python3 all-in.py process $wdir/ %path%"
