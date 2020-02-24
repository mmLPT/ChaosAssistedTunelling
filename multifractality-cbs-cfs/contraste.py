import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from scipy.interpolate import interp1d

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
	N=int(data['N'])
	i0=int(data['i0'])
	beta=data['beta']
	potential=data['potential']
	data.close()
	

	
	alpha=np.linspace(0.0,1.0,nruns)[runid]
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",i0=i0,nruns=nruns,N=N,potential=potential)

	if potential=="RS":
		pot=PotentialST(alpha)
		grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	if potential=="GG":
		pot=PotentialGG(beta,alpha*np.pi)
		grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	
	n0=1000
	wfxx0=np.zeros((n0,N,N))
	for i in range(0,n0):
		fo=CATFloquetOperator(grid,pot,randomphase=True)
		fo.diagonalize()
		for j in range(0,N):
			for k in range(0,N):
				wfxx0[i,j,k]=(np.abs(fo.getEvec(k).x[j]))**2
				
	
	Cmean=np.sum(np.mean(wfxx0**2,axis=0),axis=1)/np.sum(np.mean(wfxx0,axis=0)**2,axis=1)-1

	np.savez(wdir+"raw-data/"+str(runid),Cmean=Cmean,alpha=alpha)
	
if mode=="gather":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	N=int(data['N'])
	data.close()
	
	Cmean=np.zeros((nruns,N))
	alpha=np.zeros(nruns)
	for i in range(0,nruns):
		data=np.load(wdir+"raw-data/"+str(i)+".npz")
		Cmean[i,:]=data['Cmean']
		alpha[i]=data['alpha']
		data.close()
		
	np.savez(wdir+"final",Cmean=Cmean,alpha=alpha)
		
if mode=="plot":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	N=int(data['N'])
	data.close()
	
	data=np.load(wdir+"final.npz")
	Cmean=data['Cmean']
	alpha=data['alpha']
	data.close()
	
	
	data=np.load(wdir+"d2-final.npz")
	alpha2=data['alpha']
	D2=data['D2']
	data.close()
	
	print(alpha,alpha2)

	
	ax=plt.gca()
	ax.set_xlim(0.0,1.0)
	ax.set_ylim(0,1.0)
	
	ax.set_ylabel(r"$h_\infty=\frac{\sum_n \langle \phi_n(x_0)^4 \rangle}{\sum_n \langle \phi_n(x_0)^2 \rangle^2}$")
	ax.set_xlabel(r"a")

	
	i0=int(N/4)
	
	def Cth(x,a,b):
		return 1-(a*N)**(-x)
		

	
	popt, pcov = curve_fit(Cth, D2, Cmean[:,i0])
	
	
	plt.scatter(D2,Cmean[:,i0],c="red")
	plt.plot(D2,Cth(D2,*popt),c="blue")
	print(Cth(0,*popt))
	# ~ plt.scatter(alpha,Cmean[:,i0],c="red")

	plt.savefig(wdir+"final3.png", bbox_inches = 'tight',format="png")
	


				
	


