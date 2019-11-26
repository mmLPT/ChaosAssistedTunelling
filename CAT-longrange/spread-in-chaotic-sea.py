import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.systems.kickedrotor import *

mode=sys.argv[1]
# ~ datafile=sys.argv[2]
datafile="tempdata/spread-in-chaotic-sea-reg"

if mode=="compute":
	e=0.00000000000001
	gamma=0.15*(1-0.6)
	# ~ e=0.6
	gamma=0.15
	h=0.1
	x0=0
	tmax=100
	Ncell=51
	ncellini=1

	km=int(Ncell/2)

	# Simulation parameters
	N=Ncell*32
	itmax=int(tmax/2) # number of period simulated (*2)
	xratio=2.0

	print(N*h/Ncell)

	# Creation of objects
	pot=PotentialMP(e,gamma)
	grid=Grid(N,h,xmax=Ncell*2*np.pi)
	husimi=Husimi(grid)
	fo=CATFloquetOperator(grid,pot)
	time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
	husimi=Husimi(grid)

	### SIMULATION ###

	ind=np.arange(Ncell)
	N0=(Ncell-1)*0.5
	n0=(ncellini-1)*0.5
	indIni=np.logical_and(ind>(N0-n0-1),ind<(N0+n0+1))
	indElse=np.logical_not(indIni)

	x0t=np.linspace(x0-0.5*(Ncell-1)*2*np.pi,x0+0.5*(Ncell-1)*2*np.pi,Ncell,endpoint=True)


	wf=WaveFunction(grid)
		
	for icell in ind[indIni]:
		xicell=x0t[icell]
		wficell=WaveFunction(grid)
		wficell.setState("coherent",x0=xicell ,xratio=xratio)
		wf=wf+wficell
		
	wf.normalizeX()
	wf.x2p()



	wf0=wf.x
	prob=np.zeros((itmax,Ncell))
	xstd=np.zeros(itmax)
		
	for it in range(0,itmax):
		print(it,"/",itmax)
		
		for icell in ind:
			xicell=x0t[icell]
			wficell=WaveFunction(grid)
			wficell.setState("coherent",x0=xicell,xratio=xratio)
			prob[it,icell]=np.abs(wf%wficell)**2

		xstd[it]=wf.getxstd()
		fo.propagate(wf)
		

	np.savez(datafile,x=grid.x,wf0=wf0,wff=wf.x,prob=prob,time=time,indIni=indIni,indElse=indElse,xstd=xstd)	
	
if mode=="plot":
	data=np.load(datafile+".npz")
	x=data['x']
	wf0=data['wf0']
	wff=data['wff']
	prob=data['prob']
	xstd=data['xstd']
	time=data['time']
	indIni=data['indIni']
	indElse=data['indElse']
	data.close()

	probIni=np.sum(prob[:,indIni],axis=1)
	probElse=np.sum(prob[:,indElse],axis=1)
	
	ax=plt.gca()
	ax.set_xlim(0,30)
	ax.plot(x/(2*np.pi),np.abs(wf0)**2,c='blue')	
	ax.plot(x/(2*np.pi),np.abs(wff)**2,c='red')
	ax.set_xticks(np.arange(0,30),minor=True)
	ax.grid(which="minor")
	plt.show()

	ax=plt.gca()	
	ax.set_xlim(0,np.max(time))
	ax.plot(time,probIni,c="blue")
	ax.plot(time,probElse,c="red")
	ax.plot(time,probIni+probElse,c="green")
	ax.grid()
	
	plt.show()
	
	ax=plt.gca()	
	ax.set_xlim(0,np.max(time))
	ax.plot(time,xstd,c="blue")
	ax.grid()
	

	plt.show()



	
	




