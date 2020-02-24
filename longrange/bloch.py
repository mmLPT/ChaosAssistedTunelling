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
datafile="tempdata/bloch"




if mode=="compute":
	h=1/2.112
	e=0.5
	gamma=0.15
	
	print(1/h)

	x0=0
	tmax=1000
	Ncell=101
	ncellini=Ncell
	Npcell=32

	km=int(Ncell/2)

	# Simulation parameters
	N=Ncell*Npcell
	itmax=int(tmax/2) # number of period simulated (*2)
	xratio=2

	print(N*h/Ncell)
	
	# ~ e=0

	# Creation of objects
	pot=PotentialMP(e,gamma,Fb=-0.0001)
	grid=Grid(N,h,xmax=Ncell*2*np.pi)
	husimi=Husimi(grid)
	fo=CATFloquetOperator(grid,pot)
	time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
	husimi=Husimi(grid)

	### SIMULATION ###
	
	
	
	pot0=PotentialMP(0,gamma)
	grid0=Grid(Npcell,h,xmax=2*np.pi)
	# ~ fo0=CATFloquetOperator(grid0,pot0)
	wf0=WaveFunction(grid0)	
	data=np.load("tempdata/wf0.npz")
	wf0.x=data['wf0x']
	data.close()
	wf0.normalizeX()
	wf0.x2p()
	# ~ fo0.diagonalize()
	# ~ ind, overlaps=fo0.getOrderedOverlapsWith(wf0)
	# ~ wf0=fo0.eigenvec[ind[0]]
	# ~ print(np.abs(overlaps[0])**2)
	
	# ~ wf=WaveFunction(grid)
	# ~ wf.x=np.zeros(N,dtype=complex)
	# ~ wf.x[int((Ncell-1)/2*Npcell):int((Ncell-1)/2*Npcell+Npcell)]=wf0.x
	# ~ wf.normalizeX()
	# ~ wf.x2p()


	ind=np.arange(Ncell)
	N0=(Ncell-1)*0.5
	n0=(ncellini-1)*0.5
	indIni=np.logical_and(ind>(N0-n0-1),ind<(N0+n0+1))
	indElse=np.logical_not(indIni)

	# ~ x0t=np.linspace(x0-0.5*(Ncell-1)*2*np.pi,x0+0.5*(Ncell-1)*2*np.pi,Ncell,endpoint=True)


	wf=WaveFunction(grid)
	for icell in ind[indIni]:
		wf.x[icell*Npcell:(icell+1)*Npcell]=wf0.x*np.exp(-(icell-n0)**2/200)
	wf.normalizeX()
	wf.x2p()
	
	
	# ~ ax=plt.gca()
	# ~ ax.plot(grid.x/(2*np.pi),np.abs(wf.x)**2,c='blue')	
	# ~ plt.show()

	prob=np.zeros((itmax,Ncell))
	xstd=np.zeros(itmax)
	indcell=np.arange(-int((Ncell-1)/2),int((Ncell-1)/2+1))
	print(indcell)
		
	for it in range(0,itmax):
		print(it,"/",itmax)
		
		for icell in ind:
			wficell=WaveFunction(grid)
			wficell.x[icell*Npcell:(icell+1)*Npcell]=wf0.x
			prob[it,icell]=np.abs(wf%wficell)**2
			
		if it%10==0:
			ax=plt.gca()
			ax.plot(grid.x/(2*np.pi),np.abs(wf.x)**2,c='blue')	
			plt.show()

		xstd[it]=np.sum(indcell*prob[it])/np.sum(prob[it])
		fo.propagate(wf)
		

	np.savez(datafile,x=grid.x,wff=wf.x,prob=prob,time=time,indIni=indIni,indElse=indElse,xstd=xstd)	
	
if mode=="plot":
	data=np.load(datafile+".npz")
	x=data['x']
	wff=data['wff']
	prob=data['prob']
	xstd=data['xstd']
	time=data['time']
	indIni=data['indIni']
	indElse=data['indElse']
	data.close()

	probIni=np.sum(prob[:,indIni],axis=1)
	probElse=np.sum(prob[:,indElse],axis=1)
	
	time=time
	xstd=xstd
	fit = np.polyfit(time,xstd, 1)
	
	
	ax=plt.gca()
	# ~ ax.set_xlim(-15,15)	
	ax.plot(x/(2*np.pi),np.abs(wff)**2,c='red')
	plt.show()

	ax=plt.subplot(1,2,1)	
	ax.set_xlim(0,np.max(time))
	ax.plot(time,probIni,c="blue")
	# ~ ax.plot(time,probElse,c="red")
	# ~ ax.plot(time,probIni+probElse,c="green")

	ax=plt.subplot(1,2,2)		
	ax.set_xlim(0,np.max(time))
	ax.plot(time,xstd,c="blue")
	# ~ ax.plot(time,fit[0]*time+fit[1],c="red")
	print(fit[0])
	ax.grid()
	

	plt.show()
	
	np.savez("tempdata/time-evolution",xstd=xstd,time=time,prob=probIni)



	
	




