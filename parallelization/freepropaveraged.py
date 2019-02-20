import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
 

mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="compute":
	# Loading parameters
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")

	N=int(data['N'])
	print(N)
	description=data['description']
	e=data['e']
	x0=data['x0']
	h=data['h']
	gamma=data['gamma']
	beta0=data['beta0']
	Ndbeta=data['Ndbeta']
	dbeta=data['dbeta']
	iperiod=int(data['iperiod'])
	icheck=int(data['icheck'])
	s,nu,x0exp = modesbasic.convert2exp(gamma,h,x0)
	pot=PotentialMP(e,gamma)
	grid=Grid(N,h)
	sizet=int(iperiod/icheck)
	if x0==0.0:
		x0=pot.x0
	data.close()
	
	# Getting ID of the run
	runid=int(sys.argv[5])

	# Saving read parameters
	if runid==0:
		nruns=int(sys.argv[4])+1
		
		np.savez(wdir+"params","w", description=description, nruns=nruns, e=e,gamma=gamma,h=h,N=N,x0=x0,s=s,nu=nu,x0exp=x0exp,beta0=beta0,dbeta=dbeta,Ndbeta=Ndbeta,iperiod=iperiod,sizet=sizet)

	
	xR=np.zeros(sizet)
	xL=np.zeros(sizet)
	
	beta=np.random.normal(beta0, dbeta)
        
	fo=CATFloquetOperator(grid,pot,beta=beta)

	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)

	for i in range(0,iperiod):
		if i%icheck==0:
			xL[int(i/icheck)]=wf.getxL()
			xR[int(i/icheck)]=wf.getxR()
		fo.propagate(wf)
	
	np.savez(wdir+str(runid),"w", beta=beta, xL = xL, xR=xR)


if mode=="average":
	datain=np.load(wdir+"params.npz")
	nruns=datain['nruns']
	iperiod=datain['iperiod']
	sizet=datain['sizet']
	time=2.0*np.linspace(0.0,1.0*iperiod,num=sizet,endpoint=False)

	xRav=np.zeros(sizet)
	xLav=np.zeros(sizet)
		
	for i in range(0,nruns):
		data=np.load(wdir+str(i)+".npz")
		xR=data['xR']
		xL=data['xL']
		xRav=xRav+xR
		xLav=xLav+xL
		data.close()
			
	A=max(xRav[0],xLav[0])
	xLav=xLav/A
	xRav=xRav/A
	np.savez(wdir+"averaged-data","w",  xL = xLav, xR=xRav,time=time)

if mode=="plot":
	data=np.load(wdir+"averaged-data.npz")
	time=data['time']
	xL=data['xL']
	xR=data['xR']
	ax=plt.gca()
	ax.set_xlim(0,500.0)
	ax.set_ylim(0,1.0)
	plt.plot(time,xL, c="red")
	plt.plot(time,xR, c="blue")
	plt.show()

