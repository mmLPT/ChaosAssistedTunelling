import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *

# This scripts makes possibles to 
# 1. compute in // the spectrum of qE for differents value of [h]
# 2. gather the information
# 3. plot the spectrum

mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="compute":
	# Loading parameters
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	N=int(data['N'])
	description=data['description']
	
	e=data['e']
	x0=data['x0']
	gamma=data['gamma']
	
	beta0=data['beta0']
	Ndbeta=data['Ndbeta']
	dbeta=data['dbeta']
	
	nstates=data['nstates']
	hmin=data['hmin']
	hmax=data['hmax']
	
	
	if x0==0.0:
		x0=pot.x0
	data.close()
	
	nruns=int(sys.argv[4])+1
	runid=int(sys.argv[5])
	h=np.linspace(hmin,hmax,nruns)[runid]
	
	if runid==0:
		nruns=int(sys.argv[4])+1
		np.savez(wdir+"params","w", description=description, nruns=nruns, e=e,gamma=gamma,h=h,N=N,x0=x0,s=s,nu=nu,x0exp=x0exp,beta0=beta0,dbeta=dbeta,Ndbeta=Ndbeta,iperiod=iperiod,sizet=sizet)

	grid=Grid(N,h)
	pot=PotentialMP(e,gamma)
	
	qEs=np.zeros(nstates)
	overlaps=np.zeros(nstates)
	symX=np.zeros(nstates)
	
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=x0,xratio=2.0)
	
	fo=CATFloquetOperator(grid,pot)
	fo.diagonalize()
	fo.computeOverlapsAndQEs(wfcs)
	qEs[i],overlaps[i],symX[i]=fo.getQEsOverlapsSymmetry(nstates)
		
	np.savez(wdir+"params.npz","w", e=e,h=h, gamma=gamma, T=T, qEs=qEs, overlaps=overlaps,symX=symX, h=h, nstates=nstates)

