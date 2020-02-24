#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:45:00
#SBATCH --job-name="nu_heff"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

#g0d249-e0d14
#g0d247-e0d14
#g0d251-e0d14

nruns=179
name=g0d375-e0d24
wdir=/tmpdir/p0110mm/cat-LCAR/nu-heff-$name-${SLURM_JOB_ID}
inputfile=input/$name

python3 nu-heff-Ncell.py initialize $wdir/
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 nu-heff-Ncell.py compute $wdir/ $inputfile $nruns %path%"
python3 nu-heff-Ncell.py final $wdir/
