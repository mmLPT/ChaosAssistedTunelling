#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --job-name="MuF-Knum"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=17999
name=RS-b0d5
inputfile=input/$name
wdir=/tmpdir/p0110mm/multifractality/Knum-$name-${SLURM_JOB_ID}

python3 Knum.py initialize $wdir/ 
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 Knum.py compute $wdir/ $inputfile $nruns %path%"
python3 Knum.py average $wdir/ 
python3 Knum.py plot $wdir/ 

