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
	os.mkdir(wdir+"pic")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	
	nruns=int(sys.argv[4]) # number of runs for a given h
	runid=int(sys.argv[5])-1 # Id of the current run
	
	data=np.load(inputfile+".npz")
	alpha=data['alpha']
	N=int(data['N'])
	i0=int(data['i0'])
	tmax=int(data['tmax'])
	data.close()
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",alpha=alpha,N=N,i0=i0,tmax=tmax,nruns=nruns)
	
	wfcfs=np.zeros((3,tmax))
	K=np.zeros(tmax)
	K2=np.zeros(tmax)
	
	pot=PotentialST(alpha)
	grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	wf=WaveFunction(grid)
	wf.setState("diracx",i0=i0)
	fo=CATFloquetOperator(grid,pot,randomphase=True)
	
	
	fo.diagonalize()
	MN0=np.copy(fo.M)
	MN=np.copy(MN0)
	
	for it in range(0,tmax):
		fo.propagate(wf)
		wfcfs[0,it]=np.abs(wf.x[i0-1])**2
		wfcfs[1,it]=np.abs(wf.x[i0])**2
		wfcfs[2,it]=np.abs(wf.x[i0+1])**2
		K[it]=np.abs(np.trace(MN))**2/N
		K2[it]=fo.getFormFactor(it)
		MN=np.matmul(MN,MN0)
		
	
	np.savez(wdir+"raw-data/"+str(runid),wfcfs=wfcfs,K=K,K2=K2)
	
if mode=="average":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	tmax=int(data['tmax'])
	data.close()
	
	wfcfs=np.zeros((3,tmax))
	K=np.zeros(tmax)
	K2=np.zeros(tmax)
	
	for i in range(0,nruns):
		data=np.load(wdir+"raw-data/"+str(i)+".npz")
		wfcfs+=data['wfcfs']
		K+=data['K'] 
		K2+=data['K2'] 
		data.close()
	
	np.savez(wdir+"averaged",K=K/nruns,wfcfs=wfcfs/nruns,t=np.arange(0,tmax),K2=K2/nruns)
	
if mode=="plot":
	data=np.load(wdir+"params.npz")
	alpha=data['alpha']
	N=int(data['N'])
	i0=int(data['i0'])
	tmax=int(data['tmax'])
	nruns=int(data['nruns'])
	data.close()
	
	data=np.load(wdir+"averaged.npz")
	wfcfs=data['wfcfs']
	K=data['K']
	K2=data['K2']
	t=data['t']
	data.close()
	
	grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	wfcfs[:,0]=0.0
	
	
	ax=plt.gca()
	ax.set_xlabel(r"t")
	ax.set_xlim(0.0,tmax)
	ax.set_ylim(0.0,1.1*max(wfcfs[1,:]*(N*grid.ddx)))
	ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,alpha,nruns))
	ax.plot(t,wfcfs[0,:]*(N*grid.ddx),c="blue",zorder=0,label="iCFS-1")
	ax.plot(t,wfcfs[1,:]*(N*grid.ddx),c="green",zorder=0,label="iCFS")
	ax.plot(t,wfcfs[2,:]*(N*grid.ddx),c="red",zorder=0,label="iCFS+1")
	ax.legend()
	plt.savefig(wdir+"cfs-1.png", bbox_inches='tight',format="png")
	plt.clf()
	
	wfcfs[1,:]=wfcfs[1,:]-1.0/(N*grid.ddx)
	wfcfs[1,:]=wfcfs[1,:]/max(wfcfs[1,:])
	
	K[0]=0.0
	K=K/max(K)
	
	ax=plt.gca()
	ax.set_xlabel(r"t")
	ax.set_xlim(0.0,tmax)
	ax.set_ylim(0.0,1.1)
	ax.plot(t,wfcfs[1,:],c="blue",zorder=0,label="Height of CFS")
	ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,alpha,nruns))
	ax.scatter(t,K,c="red",zorder=1,s=1.0**2, label=r"$K(t)$")
	ax.legend()
	plt.savefig(wdir+"cfs-2.png", bbox_inches='tight',format="png")
	plt.clf()
				
	


