#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 2
#SBATCH -n 72
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=05:00:00
#SBATCH --job-name="SPS CLASS"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=1984
name=GG-a0d5-b0d1
inputfile=input/$name
wdir=/tmpdir/p0110mm/multifractality/Dq-$name-${SLURM_JOB_ID}

python3 dq.py initialize $wdir/ $inputfile
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 dq.py compute $wdir/ $inputfile $nruns %path%"
srun chdb --in-type "1 71" --command "python3 dq.py average $wdir/ %path%"
python3 dq.py gather $wdir/
python3 dq.py plot $wdir/


