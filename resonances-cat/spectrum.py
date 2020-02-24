import sys
sys.path.insert(0, '..')
import os

import numpy as np
import matplotlib.pyplot as plt

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

# State: stable [22/02/2019]

# To be used with "run-spectrum.slurm"

# This scripts makes possibles to 
# 1. compute in // the spectrum of qE for differents value of [h]
# 2. gather the information
# 3. plot the spectrum

# Arguments to provide:
# 1. mode = "compute", "gather", "plot"
# 2. working directory
# if mode=="compute":
# 	3. input file
# 	4. total number of tasks
# 	5. id of the current runs

mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"runs")

if mode=="compute":
	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	description=data['description']  
	N=int(data['N'])
	e=data['e']
	gamma=data['gamma']
	x0=data['x0'] 
	hmin=data['hmin']
	hmax=data['hmax']
	data.close()

	# Initialization of potential and correcting the x0 value if needed
	

	
	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, nruns=nruns,N=N, e=e,gamma=gamma,x0=x0,hmin=hmin,hmax=hmax)

	# Initialization of the grid for given h value
	h=1/np.linspace(1.0/hmax,1.0/hmin,nruns)[runid]
	grid=Grid(N,h)
	pot=PotentialMP(e,gamma)

	# Creating array to store data
	ind=np.zeros(N)
	qEs=np.zeros(N)
	overlaps=np.zeros(N,dtype=complex)
	symX=np.zeros(N,dtype=bool)

	# Create and diag the Floquet operator
	fo=CATFloquetOperator(grid,pot)
	fo.diagonalize()

	# Create a coherent state localized in x0 with width = 2.0 in x
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=x0,xratio=2.0)

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	ind, overlaps=fo.getOrderedOverlapsWith(wfcs)
	for i in range(0,N):
		qEs[i]=fo.getQE(ind[i])
		symX[i]=fo.getEvec(ind[i]).isSymetricInX()
		
	# Save data
	np.savez(wdir+"runs/"+str(runid),"w", h=h, qEs=qEs, overlaps=overlaps,symX=symX)

if mode=="gather":
	# This mode gather the data file from each run into gathered.npz

	# Reading inputfile
	data=np.load(wdir+"params.npz")
	nruns=data['nruns']
	hmin=data['hmin']
	hmax=data['hmax']
	N=data['N']
	e=data['e']
	gamma=data['gamma']
	data.close()

	# Create array to store data
	qEs=np.zeros((nruns,N))
	overlaps=np.zeros((nruns,N),dtype=complex)
	symX=np.zeros((nruns,N))
	h=np.zeros((nruns,N))
	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns):
		data=np.load(wdir+"runs/"+str(i)+".npz")
		qEs[i]=data['qEs']
		overlaps[i]=data['overlaps']
		symX[i]=data['symX']
		h[i]=data['h']
		data.close()

	# Save the array
	np.savez(wdir+"gathered","w", e=e,gamma=gamma,h=h,qEs=qEs,overlaps=overlaps,symX=symX,nruns=nruns)

if mode=="plot":
	# Reading inputfile
	data=np.load(wdir+"gathered.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	qEs=data['qEs']
	overlaps=np.abs(data['overlaps'])**2
	nruns=data['nruns']
	symX=data['symX']
	data.close()

	# General setup for plotting
	ax=plt.gca()
	ax.set_xlabel(r"1/h")
	ax.set_ylabel(r"$qEs T/2\pi h $")
	ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f}$".format(float(e),float(gamma)))
	ax.set_xlim(np.min(1/h),np.max(1/h))
	# ~ ax.set_xlim(2.8,3)
	ax.set_ylim(-0.5,0.5)
	ax.set_ylim(0.2,1.2)
	ax.grid(which="major",lw=1)
	ax.grid(which="minor",lw=0.1)
	
	
	# Select qEs corresponding to states that overlaps significantly with "WKB" states
	
	condOverlaps=(overlaps>0.01)
	overlaps=overlaps/np.max(overlaps)
	qEs=qEs[condOverlaps!=0]
	h=h[condOverlaps!=0]
	overlaps=overlaps[condOverlaps!=0]
	symX=symX[condOverlaps!=0]
	ind=np.argsort(overlaps) #Ordering points with overlap to have scatter to work well
	
	# Colormap for symetric and antisymetric states
	cmapSym = plt.cm.get_cmap('RdYlGn_r')
	cmapAsym = plt.cm.get_cmap('Blues')
	# ~ cmapAsym = plt.cm.get_cmap('autumn')
	# ~ cmapSym = plt.cm.get_cmap('winter')
	c=cmapSym(overlaps)
	# Filling the color of each quasi-energy with colormp propoto overlaps
	c[symX!=0]=cmapSym(overlaps[symX!=0]/np.max(overlaps[symX!=0])) 
	c[symX!=1]=cmapAsym(overlaps[symX!=1]/np.max(overlaps[symX!=1])) 
	
	phase=4*np.pi*qEs[ind]/(h[ind]*2*np.pi)
	
	ax.xaxis.set_major_locator(MultipleLocator(0.5))
	ax.yaxis.set_major_locator(MultipleLocator(0.1))
	ax.xaxis.set_minor_locator(MultipleLocator(0.02))
	ax.yaxis.set_minor_locator(MultipleLocator(0.02))

	# Plot
	plt.scatter(1/h[ind],phase,c=c[ind],s=1**2)
	plt.scatter(1/h[ind],phase+1,c=c[ind],s=1**2)
	plt.scatter(1/h[ind],phase-1,c=c[ind],s=1**2)

	plt.savefig(wdir+"spectrum.png", bbox_inches='tight') 

