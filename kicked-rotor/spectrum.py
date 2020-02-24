import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.kickedrotor import *

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
	N=int(data['N'])
	Kmin=data['Kmin']
	Kmax=data['Kmax']
	data.close()
	
	h=2*np.pi/N

	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", nruns=nruns,Kmin=Kmin,Kmax=Kmax,N=N,h=h)

	# Initialization of the grid for given h value
	K=np.linspace(Kmin,Kmax,nruns)[runid]
	grid=Grid(N,h)
	
	# Initialization of potential and correcting the x0 value if needed
	pot=PotentialKR(K)

	# Creating array to store data
	ins=np.zeros(N)
	qEs=np.zeros(N)
	overlaps=np.zeros(N,dtype=complex)
	symX=np.zeros(N,dtype=bool)

	# Create and diag the Floquet operator
	fo=CATFloquetOperator(grid,pot)
	fo.diagonalize()

	# Create a coherent state localized in x0 with width = 2.0 in x
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=0.0,xratio=2.0)

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	ind, overlaps=fo.getOrderedOverlapsWith(wfcs)
	for i in range(0,N):
		qEs[i]=fo.getQE(ind[i])
		symX[i]=fo.getEvec(ind[i]).isSymetricInX()
		
	# Save data
	np.savez(wdir+"runs/"+str(runid),"w", K=K, qEs=qEs, overlaps=overlaps,symX=symX)

if mode=="gather":
	# This mode collect the spectrum for each value of h and make a single file

	# Reading inputfile
	data=np.load(wdir+"params.npz")
	nruns=data['nruns']
	N=data['N']
	h=data['h']
	data.close()

	# Create array to store data
	qEs=np.zeros((nruns,N))
	overlaps=np.zeros((nruns,N),dtype=complex)
	symX=np.zeros((nruns,N))
	K=np.zeros((nruns,N))
	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns):
		data=np.load(wdir+"runs/"+str(i)+".npz")
		qEs[i]=data['qEs']
		overlaps[i]=data['overlaps']
		symX[i]=data['symX']
		K[i]=data['K']*np.ones(N)
		data.close()

	# Save the array
	np.savez(wdir+"gathered","w", K=K,h=h,qEs=qEs,overlaps=overlaps,symX=symX,nruns=nruns,N=N)

if mode=="plot":
	# Reading inputfile
	data=np.load(wdir+"gathered.npz")
	K=data['K']
	h=data['h']
	N=int(data['N'])
	qEs=data['qEs']
	overlaps=data['overlaps']
	nruns=data['nruns']
	symX=data['symX']
	data.close()

	# General setup for plotting
	ax=plt.gca()
	ax.set_xlabel(r"K")
	ax.set_ylabel(r"$E/h$")
	# ~ ax.set_title(r"$K={:.2f}$".format(K))
	ax.set_xlim(0.0,7.0)
	ax.set_ylim(-np.pi,np.pi)
	# ~ ax.set_ylim(-np.pi,-0.5*np.pi)
	ax.set_yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi])
	ax.set_yticklabels([r"$-\pi$",r"$-\pi/2$","0",r"$\pi/2$",r"$\pi$"])
	
	print(K)


	K1=np.extract(symX,K)
	qEs1=np.extract(symX,qEs)
	plt.scatter(K1,(qEs1/h),c="black",s=0.5**2,zorder=2)
	

	# ~ K2=np.extract(-symX+1,K)
	# ~ qEs2=np.extract(-symX+1,qEs)
	# ~ plt.scatter(K2,(qEs2/h),c="grey",s=0.5**2,zorder=2)


	plt.savefig(wdir+"spectrum.png", bbox_inches='tight',dpi=250) 
	

