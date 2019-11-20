import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.plot.latex import *
from utils.systems.modulatedpendulum import *
from utils.systems.sawtooth import *



# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data
 
mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"raw-data")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	
	nruns=int(sys.argv[4]) # number of runs for a given h
	runid=int(sys.argv[5])-1 # Id of the current run
	
	data=np.load(inputfile+".npz")
	alpha=data['alpha']
	N=int(data['N'])
	data.close()
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",alpha=alpha,N=N,nruns=nruns)

	# ~ pot=PotentialST(alpha)
	# ~ grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	pot=PotentialGG(np.pi/2.0,alpha)
	grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	# ~ pot=PotentialTR(alpha)
	# ~ grid=Grid(Ninfo[0,runid],h=2*np.pi/Ninfo[0,runid],xmax=2*np.pi)

	fo=CATFloquetOperator(grid,pot,randomphase=True)
	fo.diagonalize()
	s=fo.getSpacingDistribution()
	np.savez(wdir+"raw-data/"+str(runid),s=s)
	
if mode=="average":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	N=int(data['N'])
	data.close()
	
	s=np.array([])
	for i in range(0,nruns):
		data=np.load(wdir+"raw-data/"+str(i)+".npz")
		s=np.append(s,data['s'])
		data.close()
		
	np.savez(wdir+"averaged",s=s)
	
if mode=="plot":
	data=np.load(wdir+"averaged.npz")
	s=data['s']
	data.close()
	
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	N=int(data['N'])
	alpha=data['alpha']
	data.close()
	
	smax=5.0

	ax=plt.gca()
	ax.set_xlabel(r"s")
	ax.set_ylabel(r"P(s)")
	ax.set_xlim(0.0,smax)
	ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,alpha,nruns))
	plt.hist(s, range = (0, 1.05*smax), bins = int(nruns/10),density=True)
	plt.savefig(wdir+"sd.png", bbox_inches = 'tight',format="png")

				
	


