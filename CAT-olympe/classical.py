import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *
from utils.plot import *

# State: stable [22/02/2019]

# To be used with "run-classical.slurm"

# This scripts makes possibles to 
# 1. generate a stroboscopic trajectory
# 2. export the SPS (.png)

# Arguments to provide:
# 1. mode = "compute", "plot"
# 2. working directory
# if mode=="compute":
# 	3. input file
# 	4. total number of tasks
# 	5. id of the current runs

mode=sys.argv[1] # mode selected
wdir=sys.argv[2] # working (=output) directory

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"trajectories")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	# Description of the run
	description=data['description']

	# General physical parameters 
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	s,nu,x0exp=convert2exp(gamma,h)
	iperiod=data['iperiod']

	data.close() # close file
	
	# run info
	ny0=int(sys.argv[4]) # total number of runs
	runid=int(sys.argv[5])-1 # ID of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, ny0=ny0, e=e,gamma=gamma,h=h,iperiod=iperiod,s=s,nu=nu)

	# Create the potential, time propagator and stroboscopic phase space
	pot=PotentialMP(e,gamma)
	cp=ClassicalContinueTimePropagator(pot)
	pp=PhasePortrait(iperiod,ny0,cp,xmax=np.pi,pmax=np.pi) 

	# Generate and save a trajectory
	x,p=pp.getTrajectory(runid)
	sc=pp.getChaoticity(x,p)
	np.savez(wdir+"trajectories/"+str(runid),"w", x=x, p=p,sc=sc)
	
if mode=="gather":
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']	
	iperiod=data['iperiod']
	ny0=data['ny0']
	data.close()
	
	sc=np.zeros(ny0)
	x=np.zeros((ny0,iperiod))
	p=np.zeros((ny0,iperiod))
	norms=0.0
	for i in range(0,ny0):
		data=np.load(wdir+"trajectories/"+str(i)+".npz")
		x[i,:]=data['x']
		p[i,:] = data['p']
		sc[i]= data['sc']
		data.close()
		norms=max(sc[i],norms)
	sc=sc/norms
	np.savez(wdir+"all-trajectories","w", x=x, p=p,sc=sc)

if mode=="plot":
	# Loading inpute file
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	ny0=data['ny0']
	data.close()
	
	# General plotting setup
	ax = plt.axes()
	ax.set_xlim(-np.pi,np.pi)
	ax.set_ylim(-2.0,2.0)
	ax.set_aspect('equal')
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f}$".format(e,gamma))

	# Plotting the SPS
	cmap=plt.get_cmap("jet")

	data=np.load(wdir+"all-trajectories.npz")
	x=data["x"]
	p=data["p"]
	sc=data["sc"]
	data.close()
		
	for i in range(0,ny0):
		plt.scatter(x[i,:],p[i,:],s=0.05**2,c=cmap(sc[i]))

	plt.savefig(wdir+"phase-portrait.png", bbox_inches = 'tight',format="png")
	#plt.savefig(wdir+"phase-portrait.pdf", bbox_inches = 'tight',format="pdf")
