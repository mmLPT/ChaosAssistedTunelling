#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --job-name="SPECTRUM"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=539
name=e0d50-g0d15-hm2d416
wdir=/tmpdir/p0110mm/longrange/EBETA-${SLURM_JOB_ID}-$name
inputfile=input/$name

python3 ebeta.py initialize $wdir/
srun chdb --in-type "0 $nruns" --report "$wdir/report.txt" --command "python3 ebeta.py compute $wdir/ $inputfile $nruns %path%"
python3 ebeta.py gather $wdir/
#~ python3 spectrum.py plot $wdir/
