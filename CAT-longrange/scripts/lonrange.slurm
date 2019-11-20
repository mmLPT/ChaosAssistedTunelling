#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:20:00
#SBATCH --job-name="LRANGE"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=179
name=e0d05-g0d15
wdir=/tmpdir/p0110mm/longrange/LONGRANGE-${SLURM_JOB_ID}-$name
inputfile=input/$name

python3 long.py initialize $wdir/
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 long.py compute $wdir/ $inputfile $nruns %path%"
python3 long.py gather $wdir/
python3 long.py plot $wdir/
