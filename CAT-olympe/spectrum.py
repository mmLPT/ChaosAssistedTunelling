import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *

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
	
	nstates=int(data['nstates'])
	hmin=data['hmin']
	hmax=data['hmax']

	data.close()

	# Initialization of potential and correcting the x0 value if needed
	pot=PotentialMP(e,gamma)
	if x0==0.0:
		x0=pot.x0
	
	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, nruns=nruns,N=N, e=e,gamma=gamma,x0=x0,nstates=nstates,hmin=hmin,hmax=hmax)

	# Initialization of the grid for given h value
	h=1/np.linspace(1.0/hmax,1.0/hmin,nruns)[runid]
	grid=Grid(N,h)

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
	wfcs.setState("coherent",x0=x0,xratio=2.0)

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	ind, overlaps=fo.getOrderedOverlapsWith(wfcs)
	for i in range(0,N):
		qEs[i]=fo.getQE(ind[i])
		symX[i]=fo.getEvec(ind[i]).isSymetricInX()
		
	# Save data
	np.savez(wdir+str(runid),"w", h=h, qEs=qEs, overlaps=overlaps,symX=symX)

if mode=="gather":
	# This mode collect the spectrum for each value of h and make a single file

	# Reading inputfile
	data=np.load(wdir+"params.npz")
	nruns=data['nruns']
	nstates=data['nstates']
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
	h=1/np.linspace(1.0/hmax,1.0/hmin,nruns)
	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns):
		data=np.load(wdir+str(i)+".npz")
		qEs[i]=data['qEs']
		overlaps[i]=data['overlaps']
		symX[i]=data['symX']
		data.close()

	# Save the array
	np.savez(wdir+"gathered","w", e=e,gamma=gamma,h=h,qEs=qEs,overlaps=overlaps,symX=symX,nstates=nstates,nruns=nruns)

if mode=="plot":
	# Reading inputfile
	data=np.load(wdir+"gathered.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	qEs=data['qEs']
	overlaps=data['overlaps']
	nstates=data['nstates']
	nruns=data['nruns']
	symX=data['symX']
	data.close()

	# General setup for plotting
	ax=plt.gca()
	ax.set_xlabel(r"h")
	ax.set_ylabel(r"$qEs/h$")
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.2f}$".format(e,gamma))
	#ax.set_xlim(min(1/h),max(1/h))
	#ax.set_ylim(-1.0,3.0)

	for irun in range(0,nruns):
		# We want to rescale the overlap to 1 for each value of h
		# to do so we start focusing on the 2 states with highest overlap
		# check if they are sym or asym
		if symX[irun,0]==True:
			Nsym=abs(overlaps[irun,0])**2
			Nasym=abs(overlaps[irun,1])**2
		else:
			Nsym=abs(overlaps[irun,1])**2
			Nasym=abs(overlaps[irun,0])**2

		# Then we create colormap
		cmapSym = plt.cm.get_cmap('Blues')
		cmapAsym = plt.cm.get_cmap('Reds')

		# And generate on of lenght nstates scaled by weight for sym and asym
		# so that it goes from 0 to 1
		rgbaSym = cmapSym(abs(overlaps[irun,:])**2/Nsym)
		rgbaAsym = cmapAsym(abs(overlaps[irun,:])**2/Nasym)
		#print(irun)

		N=5
		for istate in range(0,N):
			if symX[irun,istate]==True:
				plt.scatter(1/h[irun],2*np.pi*qEs[irun,istate]/h[irun],c=rgbaSym[istate],s=0.5**2,zorder=N-istate)
			else:
				plt.scatter(1/h[irun],2*np.pi*qEs[irun,istate]/h[irun],c=rgbaAsym[istate],s=0.5**2,zorder=N-istate)


	plt.show()
	#plt.savefig(wdir+"spectrum.pdf", bbox_inches='tight',format="pdf") 
	

