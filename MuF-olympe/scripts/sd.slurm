#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --job-name="MuF-SD"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=10000
name=a_0d7
inputfile=input/$name
wdir=/tmpdir/p0110mm/MuF-SpacingDistribution-$name-${SLURM_JOB_ID}

python3 sd.py initialize $wdir/ $inputfile
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 sd.py compute $wdir/ $inputfile $nruns %path%"
python3 sd.py average $wdir/ 
python3 sd.py plot $wdir/ 


