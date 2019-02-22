import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *

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
	sb=StrobosopicPhaseSpace(iperiod,ny0,cp)

	# Generate and save a trajectory
	xs,ps = sb.getTrajectory(runid)
	np.savez(wdir+str(runid),"w", x=xs, p=ps)

if mode=="plot":
	# Loading inpute file
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']	
	iperiod=data['iperiod']
	s=data['s']
	nu=data['nu']
	ny0=data['ny0']
	data.close()
	
	# General plotting setup
	ax = plt.axes()
	ax.set_xlim(-np.pi,np.pi)
	ax.set_ylim(-1.0,1.0)
	ax.set_aspect('equal')
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.2f} \quad s={:.2f} \quad \nu={:.1f} kHz$".format(e,gamma,s,nu/1000.0))

	# Plotting the SPS
	for i in range(0,ny0):
		data=np.load(wdir+str(i)+".npz")
		x=data["x"]
		p=data["p"]
		plt.scatter(x,p,s=01.0**2)
	
	# Export the SPS as .png
	plt.savefig(wdir+"SPS.png")
