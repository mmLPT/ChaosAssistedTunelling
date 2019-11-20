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
name=GG-b0d1
inputfile=input/$name
wdir=/tmpdir/p0110mm/multifractality/D2-$name-${SLURM_JOB_ID}

python3 d2.py initialize $wdir/ $inputfile
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 d2.py compute $wdir/ $inputfile $nruns %path%"
srun chdb --in-type "1 71" --command "python3 d2.py average $wdir/ %path%"
python3 d2.py gather $wdir/
python3 d2.py plot $wdir/





