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

# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data
 
mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	ne=int(sys.argv[3])
	ngamma=int(sys.argv[4])
	os.mkdir(wdir)
	os.mkdir(wdir+"pictures")
	for ih in range(0,ne*ngamma):
		os.mkdir(wdir+str(ih))

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	# Description of the run
	description=data['description']

	# General physical parameters 
	N=int(data['N'])
	x0=data['x0']
	iperiod=int(data['iperiod']) #number of period

	# Evaluated parameters
	hmin=data['hmin']
	hmax=data['hmax']
	emin=data['emin']
	emax=data['emax']
	gammamin=data['gammamin']
	gammamax=data['gammamax']
	
	data.close() # close the input file
	
	
	# Management of the ID
	nruns=int(sys.argv[4]) # Total number of // runs
	ne=int(sys.argv[5]) # number of runs for a given h
	ngamma=int(sys.argv[6]) # number of runs for a given h
	nparams=ne*ngamma
	runid=int(sys.argv[7])-1 # Id of the current run
	nh=int(nruns/nparams)
	indexparams=int(runid/nh)
	indexh=runid%nh


	# Creation of paramters table corresponding to ID
	gammat=np.linspace(gammamin,gammamax,ngamma)
	et=np.linspace(emin,emax,ne)
	ht=np.linspace(hmin,hmax,nh)
	params=np.zeros((nparams,2))
	for ie in range(0,ne):
		for igamma in range(0,ngamma):
			params[igamma+ie*ngamma]=np.array((et[ie],gammat[igamma]))

	# Create array to store "Left" and "Right" observables
	xR=np.zeros(iperiod)
	xL=np.zeros(iperiod)

	# Initialization of the grid for given h value
	e,gamma=params[indexparams]
	h=ht[indexh]
	grid=Grid(N,h)
	pot=PotentialMP(e,gamma)
	if x0==0.0:
		x0=pot.x0

	# Create the Floquet operator
	fo=CATFloquetOperator(grid,pot)

	# Create the initial state: a coherent state localized in x0 with width = 2.0 in x
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)

	# Propagate the wavefunction over iperiod storing the observable every time
	for i in range(0,iperiod):
		xL[i]=wf.getxL()
		xR[i]=wf.getxR()
		fo.propagate(wf)

	np.savez(wdir+str(indexparams)+"/"+str(indexh),"w", xL = xL, xR=xR)


	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description,N=N, params=params, h=ht ,e=et,gamma=gammat,x0=x0,ne=ne,ngamma=ngamma,nparams=nparams,nh=nh, nruns=nruns,iperiod=iperiod)


if mode=="process":
	runid=int(sys.argv[3])-1 # Id of the current run

	data=np.load(wdir+"params.npz")
	nh=data['nh']
	iperiod=data['iperiod']
	h=data['h']
	params=data['params']
	x0=data['x0']

	data.close()

	iTF=100 
	#iTF=int(iperiod)

	density=np.zeros((int(iTF/2)+1,nh))
	omegas=np.fft.rfftfreq(iTF,d=2.0)*2*np.pi
	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	e,gamma=params[runid]

	for ih in range(0,nh):
		data=np.load(wdir+str(runid)+"/"+str(ih)+".npz")
		xL=data['xL'][:iTF].copy()
		xR=data['xR'][:iTF].copy()
		xLf=np.abs(np.fft.rfft(xL))
		xRf=np.abs(np.fft.rfft(xR))
		xLf[0]=0.0
		xRf[0]=0.0
		density[:,ih]=(xLf+xRf)*0.5

	a=np.percentile(density,99.80)
	density=density/a

	ax=plt.gca()
	
	ax.set_xlabel("h")
	ax.set_ylabel("omega")
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f}$".format(e,gamma))

	#ax.set_xlim(min(h),max(h))
	#ax.set_ylim(0.001,max(omegas))
	#ax.set_yscale("log")

	omegas[0]=omegas[1]
	
	levels = MaxNLocator(nbins=100).tick_values(0.0,1.0)	
	cmap = plt.get_cmap('Greys')
	#norm = colors.LogNorm(0.01,1.0) 
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	#plt.contourf(h,omegas,np.sqrt(density), levels=levels,cmap=cmap)
	plt.pcolormesh(h,omegas,density, norm=norm,cmap=cmap)

	plt.savefig(wdir+"pictures/"+strint(runid)+".png")
	#plt.show()



	


