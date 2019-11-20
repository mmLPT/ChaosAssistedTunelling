#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 3
#SBATCH -n 108
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:10:00
#SBATCH --job-name="SPP"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=107
name=K10d0
wdir=/tmpdir/p0110mm/kicked-rotor/SPP-${SLURM_JOB_ID}-$name
inputfile=input/$name

python3 classical.py initialize $wdir/ $nruns
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 classical.py compute $wdir/ $inputfile $nruns %path%"
python3 classical.py gather $wdir/ $nruns
python3 classical.py plot $wdir/ $nruns


