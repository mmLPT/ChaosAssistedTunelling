import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.sawtooth import *
from scipy.optimize import curve_fit



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
	potential=data['potential']
	alpha=data['alpha']
	beta=data['beta']
	N=int(data['N'])
	
	i0=int(data['i0'])
	atNmax=int(data['atNmax'])
	tcheck=int(data['tcheck'])
	
	data.close()
	
	if potential=="RS":
		pot=PotentialST(alpha)
		grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
		
	if potential=="GG":
		pot=PotentialGG(beta,alpha*2*np.pi)
		grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
		
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",alpha=alpha,N=N,i0=i0,atNmax=atNmax,nruns=nruns,tcheck=tcheck,beta=beta,potential=potential)
		ax=plt.gca()
		ax.set_xlabel(r"x")
		ax.set_ylabel(r"V(x)")
		ax.plot(grid.x,pot.Vx(grid.x),c="blue")
		plt.savefig(wdir+"pot.png", bbox_inches='tight',format="png")
		plt.clf()
		
	K=np.zeros(tcheck)
	wfxx0=np.zeros(N)
	
	fo=CATFloquetOperator(grid,pot,randomphase=True)
	fo.diagonalize()
	
	for i in range(0,N):
		wfxx0[i]=(np.abs(fo.getEvec(i).x[i0]))**2
		
	for i in range(0,tcheck):
		K[i]=fo.getFormFactor(i*atNmax/alpha*N/tcheck)

	np.savez(wdir+"raw-data/"+str(runid),K=K,wfxx0=wfxx0)
	
if mode=="average":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	atNmax=int(data['atNmax'])
	tcheck=int(data['tcheck'])
	N=int(data['N'])
	i0=int(data['i0'])
	data.close()
	
	
	Kt=np.zeros((nruns,tcheck))
	wfxx0t=np.zeros((nruns,N))
	
	for i in range(0,nruns):
		data=np.load(wdir+"raw-data/"+str(i)+".npz")
		Kt[i,:]=data['K'] 
		wfxx0t[i,:]=data['wfxx0'] 
		data.close()
		
	K=np.mean(Kt,axis=0)
	Cinf=np.sum(np.mean(wfxx0t**2,axis=0))/np.sum(np.mean(wfxx0t,axis=0)**2)-1
	time=np.linspace(0.0,atNmax,tcheck)
	
	K[0]=0.0
	
	np.savez(wdir+"averaged",K=K,Cinf=Cinf,time=time)
	
if mode=="plot":
	data=np.load(wdir+"averaged.npz")
	time=data['time']
	K=data['K']
	Cinf=data['Cinf']
	data.close()
	
	ax=plt.gca()
	ax.set_xlim(0.0,10.0)
	ax.set_ylim(0.0,1.75)
	ax.plot(time,K*Cinf)
	plt.savefig(wdir+"Knum.png", bbox_inches = 'tight',format="png")

				
	


