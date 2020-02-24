import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *
from scipy.linalg import logm, expm

mode=sys.argv[1]


tmax=1500
itmax=int(tmax/2)
h=1/2.112
e=0.5
gamma=0.15

Ncell=9
Npcell=32
N=Npcell*Ncell
nstates=Ncell

print(N*h/Ncell)

if mode=="diag":

	pot=PotentialMP(e,gamma)
	grid=Grid(N,h,xmax=Ncell*2*np.pi)
	fo=CATFloquetOperator(grid,pot)
	fo.diagonalize()

	wf=WaveFunction(grid)
	wf.setState("coherent",x0=0.0,xratio=2.0)	

	ind, overlaps=fo.getOrderedOverlapsWith(wf)

	qE=np.zeros(Ncell)
	beta=np.zeros(Ncell)
	for i in range(0,Ncell):
		qE[i]=fo.qE[ind[i]]
		wfx=fo.eigenvec[ind[i]].x
		wfxt=np.roll(wfx,Npcell)
		beta[i]=np.mean(np.angle(wfxt/wfx))
		print(beta[i],np.abs(overlaps[i])**2)

	np.savez("tempdata/states",beta=beta,qEs=qE,h=h)
	
if mode=="tb":

	data=np.load("tempdata/states.npz")
	beta=data['beta']
	qEs=data['qEs']
	h=data['h']
	data.close()
	
	data=np.load("tempdata/gathered.npz")
	beta0=data['beta']
	qEs0=data['qEs']
	h0=data['h']
	data.close()

	ax=plt.gca()
	ax.set_xlim(-np.pi,np.pi)
	ax.set_xlabel("beta")
	ax.set_ylabel("quasi-energies")
	ax.scatter(beta0*2*np.pi/h0,qEs0,s=0.2,c="red",label="Monocellule avec quasi-moment")
	ax.scatter(beta,qEs,c="blue",label="9 cellules")
	ax.legend()
	plt.show()

	Ncell=qEs.size

	qEs=qEs[np.argsort(beta)]
	V=np.fft.rfft(qEs)/(Ncell)
		
	H=np.zeros((Ncell,Ncell),dtype=complex)

	# ~ def d(i,j):
		# ~ if np.abs(i-j)<int(0.5*(Ncell-1)):
			# ~ return np.abs(i-j)
		# ~ else:
			# ~ return Ncell-1-np.abs(i-j)
		
	for i in range(0,Ncell):
		for j in range(i,Ncell):
			# ~ print(i,j,d(i,j))
			if np.abs(i-j)<=int(0.5*(Ncell-1)):
				H[i,j]=V[np.abs(i-j)]
			else:
				H[i,j]=np.conjugate(V[Ncell-np.abs(i-j)])
			H[j,i]=np.conjugate(H[i,j])
		print(np.abs(H[i])**2)
				

	U=expm(-1j*H*4*np.pi/h)
	X=np.zeros(Ncell)
	n0=int(0.5*(Ncell-1))
	X[n0]=1

	prob=np.zeros((itmax,Ncell),dtype=complex)
	xstd=np.zeros(itmax)
	time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
	indcell=np.arange(Ncell)
	
	

	for it in range(0,itmax):
		prob[it]=X
		xstd[it]=np.sqrt(np.sum((np.arange(Ncell)-n0)**2*np.abs(prob[it])**2)/np.sum(np.abs(prob[it])**2))
		X=np.matmul(U,X)
		
		
	data=np.load("tempdata/time-evolution.npz")
	xstd_te=data['xstd']
	time_te=data['time']
	prob_te=data['prob']
		
	ax=plt.subplot(1,2,1)
	ax.scatter(time,xstd,c="red",zorder=0,label="Modèle tight-binding")
	ax.plot(time_te,xstd_te,c="blue",zorder=1,lw=3,label="Propagation temporelle")
	ax.set_xlim(0,np.max(time))
	ax.set_xlabel(r"Time")
	ax.set_ylabel(r"sqrt(<x^2>)")
	ax.legend()
	
	ax=plt.subplot(1,2,2)
	ax.scatter(time,np.abs(prob[:,n0])**2,c="red",zorder=0,label="Modèle tight-binding")
	ax.plot(time_te,prob_te,c="blue",zorder=1,lw=3,label="Propagation temporelle")
	ax.set_xlim(0,np.max(time))
	ax.legend()
	ax.set_xlabel("Time")
	ax.set_ylabel("probabilite pic initial")
	plt.show()










