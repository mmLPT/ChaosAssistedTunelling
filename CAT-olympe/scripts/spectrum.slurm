#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:10:00
#SBATCH --job-name="SPECTRUM"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=359
name=e0d59-g0d225
wdir=/tmpdir/p0110mm/cat-LCAR/SPECTRUM-${SLURM_JOB_ID}-$name
inputfile=input/$name

python3 spectrum.py initialize $wdir/
srun chdb --in-type "0 $nruns" --report "$wdir/report.txt" --command "python3 spectrum.py compute $wdir/ $inputfile $nruns %path%"
python3 spectrum.py gather $wdir/
python3 spectrum.py plot $wdir/
