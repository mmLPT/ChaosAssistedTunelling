#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=05:00:00
#SBATCH --job-name="ALLIN COMP"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=18000
nh=200
name=allin1
wdir=data/output/$name-allin-${SLURM_JOB_ID}
inputfile=data/input/$name

python3 all-in.py initialize $wdir/ $nh
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 all-in.py compute $wdir/ $inputfile $nruns $nh %path%"
#srun chdb --in-type "1 $nh" --command "python3 all-in.py process $wdir/ %path%"
# mpirun -n 3 chdb --in-type "1 500" --command "python3 all-in.py process data/output/allin1-124813/ %path%"

