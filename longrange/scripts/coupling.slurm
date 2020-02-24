#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:10:00
#SBATCH --job-name="COUPLING"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=71
nruns2=35
name=e0d0-g0d15-hm1d66
wdir=/tmpdir/p0110mm/cat-LCAR/COUPLING-${SLURM_JOB_ID}-$name
inputfile=input/$name

python3 coupling.py initialize $wdir/
srun chdb --in-type "0 $nruns" --report "$wdir/report.txt" --command "python3 coupling.py compute $wdir/ $inputfile $nruns %path%"
python3 coupling.py gather $wdir/
srun chdb --in-type "0 $nruns2" --report "$wdir/report.txt" --command "python3 coupling.py compute2 $wdir/ $inputfile $nruns2 %path%"
python3 coupling.py gather2 $wdir/
#~ python3 long.py plot $wdir/
