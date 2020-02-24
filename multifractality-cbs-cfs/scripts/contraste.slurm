#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 2
#SBATCH -n 72
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --job-name="MuF-CFS"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=71
name=GG-b0d25
inputfile=input/$name
wdir=/tmpdir/p0110mm/multifractality/Contraste-$name-${SLURM_JOB_ID}

python3 contraste.py initialize $wdir/ 
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 contraste.py compute $wdir/ $inputfile $nruns %path%"
python3 contraste.py gather $wdir/ 
python3 contraste.py plot $wdir/ 

