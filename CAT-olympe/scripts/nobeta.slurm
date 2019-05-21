#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:15:00
#SBATCH --job-name="no beta"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=180
name=e010-g055
wdir=/tmpdir/p0110mm/$name-nobeta-${SLURM_JOB_ID}
inputfile=input/$name

python3 nobeta.py initialize $wdir/
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 nobeta.py compute $wdir/ $inputfile $nruns %path%"
python3 nobeta.py final $wdir/
