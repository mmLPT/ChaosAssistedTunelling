#!/bin/bash
#SBATCH -J chbd
#SBATCH -N 5
#SBATCH -n 180
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --job-name="MuF-diff"


module purge
module load openmpi/gnu/2.0.2.10 chdb/1.0-ompi python/3.6.3
ulimit -s 10240

nruns=17900
nruns2=179
name=GG-a0d1-b0d1
inputfile=input/$name
wdir=/tmpdir/p0110mm/MuF-diffusion-$name-${SLURM_JOB_ID}

python3 diffusion.py initialize $wdir/ $inputfile
srun chdb --in-type "1 $nruns" --report "$wdir/report.txt" --command "python3 diffusion.py compute $wdir/ $inputfile $nruns %path%"
srun chdb --in-type "1 $nruns2" --report "$wdir/report.txt" --command "python3 diffusion.py average $wdir/ $nruns2 %path%"
python3 diffusion.py average2 $wdir/
python3 diffusion.py plot $wdir/


