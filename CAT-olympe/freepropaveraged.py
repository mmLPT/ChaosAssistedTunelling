import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *

# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data

# Arguments to provide:
# 1. mode = "compute", "average", "plot"
# 2. working directory
# if mode=="compute":
# 	3. input file
# 	4. total number of tasks
# 	5. id of the current runs
 
mode=sys.argv[1] # mode selected
wdir=sys.argv[2] # working (=output) directory

if mode=="compute":
	# Compute a single free propagation for a given value of quasimomentum
	# Loading parameters
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	# General information
	description=data['description']
	
	# Hamiltonian parameters
	e=data['e']
	h=data['h']
	gamma=data['gamma']
	
	# Quasimomentum distribution
	beta0=data['beta0'] # average
	Ndbeta=data['Ndbeta'] # number of cell initially ocuppied by the true system
	
	# Other
	N=int(data['N']) # number of points for the grid
	iperiod=int(data['iperiod']) # number of period
	x0=data['x0'] # initial position
	data.close()

	# Adjust value of x0 + generate exprimental parameters
	pot=PotentialMP(e,gamma)
	s,nu,x0exp = convert2exp(gamma,h,x0)
	
	# Getting ID of the run
	runid=int(sys.argv[5])-1

	# Saving read parameters
	if runid==0:
		nruns=int(sys.argv[4]) # total number of runs
		np.savez(wdir+"params","w", description=description, nruns=nruns, e=e,gamma=gamma,h=h,N=N,x0=x0,s=s,nu=nu,x0exp=x0exp,beta0=beta0,Ndbeta=Ndbeta,iperiod=iperiod)

	# Create the grid
	grid=Grid(N,h)
	
	# Create the array to store the observables left/right
	xR=np.zeros(iperiod)
	xL=np.zeros(iperiod)
	xM=np.zeros(iperiod)
	xexp=np.zeros(iperiod)
	
	# Generate quasimomentum value
	dbeta=h/(3*Ndbeta)
	beta=np.random.normal(beta0, dbeta)
        
	# Create the Floquet operator
	fo=CATFloquetOperator(grid,pot,beta=beta)

	# Create the initial state: coherent localized in x0, with width 2.0 in x direction
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)

	# Propagate the wavefunction over iperiods
	#xm=0.5*np.pi
	#xm=0.2
	xm=0.0
	for i in range(0,iperiod):
		xL[i]=wf.getxM(-np.pi,-xm)
		xR[i]=wf.getxM(xm,np.pi)
		if xm > 0.0:
			xM[i]=wf.getxM(-xm,xm)
		xexp[i]=wf.getx()
		fo.propagate(wf)

	
	# Save the observables
	np.savez(wdir+str(runid),"w", beta=beta, xL = xL, xR=xR, xM=xM,xexp=xexp)


if mode=="average":
	# This mode collect and average the observables right/left

	# Loading input file
	datain=np.load(wdir+"params.npz")
	nruns=datain['nruns']
	iperiod=datain['iperiod']

	# Generate a single time array from iperiod
	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)

	# Create array to store averaged observables
	xRav=np.zeros(iperiod)
	xLav=np.zeros(iperiod)
	xMav=np.zeros(iperiod)
	xexpav=np.zeros(iperiod)
		
	# Collect and average observables over nruns files
	for i in range(0,nruns):
		data=np.load(wdir+str(i)+".npz")
		xR=data['xR']
		xL=data['xL']
		xM=data['xM']
		xexp=data['xexp']
		xRav=xRav+xR
		xLav=xLav+xL
		xMav=xMav+xM
		xexpav=xexpav+xexp
		data.close()
	
	# Normalization	
	A=xRav[0]+xLav[0]+xMav[0]
	xLav=xLav/A
	xRav=xRav/A
	xMav=xMav/A
	xexpav=xexpav/nruns

	# Save the data
	np.savez(wdir+"averaged-data","w",  xL = xLav, xR=xRav, xM=xMav, xexp=xexpav, time=time)

if mode=="plot":
	# This mode plot averaged observables

	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	x0=data['x0']
	s=data['s']
	nu=data['nu']
	x0exp=data['x0exp']
	Ndbeta=data['Ndbeta']
	beta0=data['beta0']/h
	
	data.close()

	# Loading file
	data=np.load(wdir+"averaged-data.npz")
	time=data['time']
	xL=data['xL']
	xR=data['xR']
	xM=data['xM']
	xexp=data['xexp']

	# Plotting setup
	ax=plt.gca()
	ax.set_xlim(0,max(time))
	ax.set_ylim(0,1.0)

	# Plot
	plt.plot(time,xL, c="red")
	plt.plot(time,xR, c="blue")
	plt.plot(time,xM, c="orange")

	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f} \quad h={:.3f} \quad Ncells={:.0f} \quad x_0={:.1f} \quad \beta_0={:.2f}$".format(e,gamma,h,Ndbeta,x0,beta0)+"\n"+ r"$s={:.3f} \quad nu={:.2f} \ kHz \quad x_0={:.1f}$".format(s,nu/1000,x0exp))
	plt.savefig(wdir+"freepop.png") # exporting figure as png

	ax.clear()

	ax=plt.gca()
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f} \quad h={:.3f} \quad Ncells={:.0f} \quad x_0={:.1f} \quad \beta_0={:.2f}$".format(e,gamma,h,Ndbeta,x0,beta0)+"\n"+ r"$s={:.3f} \quad nu={:.2f} \ kHz \quad x_0={:.1f}$".format(s,nu/1000,x0exp))
	ax.set_xlim(0,max(time))
	ax.set_ylim(-np.pi/2.0,np.pi/2.0)

	# Plot
	plt.plot(time,xexp)
	plt.savefig(wdir+"freepop2.png")

