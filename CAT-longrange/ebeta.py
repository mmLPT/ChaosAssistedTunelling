import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.toolsbox import *

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from scipy.linalg import expm, sinm, cosm
from utils.mathtools.periodicfunctions import *
import matplotlib as mpl

from hamTB import *

# ==================================================================== #
# Ce script permet de calculer le spectre de l'opérateur de Floquet en 
# fonction du quasimoment beta
# Il permet ensuite de reconstruire un Hamiltonien effectif
# ==================================================================== 


mode=sys.argv[1]
wdir=sys.argv[2]


if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"runs")

if mode=="compute":
	
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	N=int(data['N'])
	e=data['e']
	gamma=data['gamma']
	x0=data['x0'] 
	h=data['h']
	data.close()

	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
	if runid==0: 	
		np.savez(wdir+"params","w", nruns=nruns,N=N, e=e,gamma=gamma,x0=x0,h=h)

	# Create and diag the Floquet operator
	grid=Grid(N,h)
	
	b0=np.linspace(0,1,nruns,endpoint=False)
	beta=np.sort(b0*(b0<=0.5)+(b0-1)*(b0>0.5))[runid]*h
	
	# ~ b0=np.linspace(-0.5,0.5,nruns,endpoint=False)
	# ~ beta=b0[runid]*h
	
	
	pot=PotentialMP(e,gamma)
	fo=CATFloquetOperator(grid,pot,beta=beta)
	fo.diagonalize()

	# Create a coherent state localized in x0 with width = 2.0 in x
	wfcs0=WaveFunction(grid)
	wfcs0.setState("coherent",x0=x0,xratio=2.0)
	pot0=PotentialMP(0,gamma)
	fo0=CATFloquetOperator(grid,pot0,beta=beta)
	fo0.diagonalize()
	ind0, overlaps0=fo0.getOrderedOverlapsWith(wfcs0)	
	wfcs0=fo0.eigenvec[ind0[0]]

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	ind, overlaps=fo.getOrderedOverlapsWith(wfcs0)
	ind=np.flipud(ind)
	overlaps=np.flipud(overlaps)

	symX=np.zeros(N)
	qEs=np.zeros(N)
	evecx=np.zeros(N,dtype=complex)
	evecp=np.zeros(N,dtype=complex)
	for i in range(0,ind.size):
		qEs[i]=fo.qE[ind[i]]
		symX[i]=fo.eigenvec[ind[i]].isSymetricInX()
		
	evecx=fo.eigenvec[ind[-1]].x
	evecp=fo.eigenvec[ind[-1]].p
		
	# Save data
	np.savez(wdir+"runs/"+str(runid),"w", qEs=qEs, overlaps=overlaps,symX=symX,beta=beta,evecx=evecx,evecp=evecp)

if mode=="gather":
	# This mode collect the spectrum for each value of h and make a single file

	# Reading inputfile
	data=np.load(wdir+"params.npz")
	nruns=data['nruns']
	h=data['h']
	N=int(data['N'])
	e=data['e']
	gamma=data['gamma']
	data.close()

	# Create array to store data
	qEs=np.zeros((nruns,N))
	overlaps=np.zeros((nruns,N),dtype=complex)
	symX=np.zeros((nruns,N))
	beta=np.zeros((nruns,N))
	evecx=np.zeros((nruns,N),dtype=complex)
	evecp=np.zeros((nruns,N),dtype=complex)
	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns):
		data=np.load(wdir+"runs/"+str(i)+".npz")
		qEs[i]=data['qEs']
		beta[i]=data['beta']
		overlaps[i]=data['overlaps']
		symX[i]=data['symX']
		evecx[i]=data['evecx']
		evecp[i]=data['evecp']
		data.close()

	# Save the array
	np.savez(wdir+"gathered","w", e=e,gamma=gamma,h=h,qEs=qEs,overlaps=overlaps,symX=symX,nruns=nruns,beta=beta,evecx=evecx,evecp=evecp)

if mode=="plot":
	# Reading inputfile
	data=np.load(wdir+"gathered.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	beta=data['beta']
	qEs=data['qEs']
	overlaps=data['overlaps']
	nruns=int(data['nruns'])
	symX=data['symX']
	evecx=data['evecx']
	data.close()
	
	
	beta=beta[:,-1]/h*2*np.pi # Renormalisation de beta
	qEs=qEs[:,-1] # Selection de l'état qui overlap le plus
	
	# ~ beta=np.roll(beta,int((beta.size+1)/2))
	# ~ beta=beta*(beta>=0)+(beta+2*np.pi)*(beta<0)

	
	# Plot
	ax=plt.gca()
	
	
	ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f} \quad 1/h={:.3f} \quad N_p={:d}$".format(float(e),float(gamma),float(1/h),int(beta.size)))
	ax.set_xlabel(r"$\beta$")
	ax.set_ylabel(r"$qE/h$")
	ax.set_xlim(np.min(beta),np.max(beta))	

	ax.plot(beta,qEs-np.mean(qEs),c="blue",zorder=1)
	ax.scatter(beta[ind0],qEs[ind0]-np.mean(qEs),c="red",s=5**2,zorder=2)
	plt.savefig(wdir+"spectrum.png", bbox_inches='tight',dpi=250)
	
	
	np.savez("tempdata/bspectrum",beta=beta,qEs=qEs-qEs[-1])
	
	
if mode=="tb-extract":
	
	dataTB=sys.argv[2]

	tb=TightBinding(dataTB)
	
	ax=plt.gca()
	ax.set_yscale('log')
	ax.set_xlim(0,200)
	ax.set_ylim(10**(-8),10**(-3))
	
	Vn=np.delete(tb.Vn,0)
	
	np.savez("tempdata/VnREG",Vn=np.abs(Vn),n=np.arange(Vn.size)+1)
	
	plt.scatter(np.arange(Vn.size),np.abs(Vn))
	
	index = (np.diff(np.sign(np.diff(np.abs(Vn)))) < 0).nonzero()[0]+1
	plt.plot(np.arange(Vn.size)[index],np.abs(Vn)[index],c="red")
	
	plt.show()
	
	
if mode=="tb-comparaison":
	
	dataTB=sys.argv[2]

	tb=TightBinding(dataTB)
	
	tb.Vnth()

	
	plt.show()


	
if mode=="tb":
	# Reading inputfile

	
	itmax=5000
	
	Nred= 0
	tb=TightBinding(wdir,Nred)
	U=tb.U
	Ncell=tb.Ncell
	
	
	# ket
	n0=int(0.5*(Ncell-1))
	tb.wf[n0]=1
	
	
	# Observables
	prob=np.zeros((itmax,Ncell))
	xstd=np.zeros(itmax)
	time=2*np.linspace(0.0,itmax,num=itmax,endpoint=False)

	# Propagation
	for it in range(0,itmax):
		# ~ print(it)
		prob[it]=np.abs(tb.wf)**2
		xstd[it]=tb.xstd()
		tb.propagate()
		
		# ~ print(np.sum(prob[it]))
		
	# ~ print(np.arange(Ncell)-n0)
	
	# Plot
	ax=plt.subplot(1,2,1)

	ax.scatter(time,np.abs(prob[:,n0])**2,c="red",label="P(n0,t)")
	ax.scatter(time,np.abs(prob[:,n0+1])**2,c="blue",label="P(n0+1,t)")
	ax.plot(time,np.abs(prob[:,n0-1])**2,c="green",label="P(n0-1,t)")
	
	ax.legend()
	ax.set_xlabel("time")
	ax.set_ylabel("probabilité")
	
	ax=plt.subplot(1,2,2)

	ax.scatter(time,xstd,c="red",label="Modèle tight-binding")
	
	ax.legend()
	ax.set_xlabel("time")
	ax.set_ylabel("sqrt(<x^2>)")
	
	plt.show()
	
	np.savez(wdir+"TB",time=time,xstd=xstd,prob=prob,n0=n0)

	plt.savefig(wdir+"TB2.png", bbox_inches='tight',dpi=250)
	
if mode=="bloch":
	# Reading inputfile
	data=np.load(wdir+"gathered.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	beta=data['beta']
	qEs=data['qEs']
	overlaps=data['overlaps']
	nruns=int(data['nruns'])
	symX=data['symX']
	data.close()
	
	F=0.0001
	
	print(1/h)
	
	
	Ncell=nruns
	itmax=2500

	
	H=HamTB()
	n0=int(0.5*(Ncell-1))
	for i in range(0,Ncell):
		H[i,i]=H[i,i]+F*(i-n0)
	U=expm(-1j*H*4*np.pi/h)
	
	
	# ~ U=0.5*(U+np.transpose(U))
	
	X=np.zeros(Ncell,dtype=complex)
	X0=np.arange(Ncell)
	X=np.exp(-(X0-n0)**2/250)
	X=X/np.sum(X**2)
	ax=plt.gca()
	ax.scatter(np.arange(Ncell),np.abs(X)**2)
	plt.show()
	


	prob=np.zeros((itmax,Ncell),dtype=complex)
	xm=np.zeros(itmax)
	time=np.linspace(0.0,itmax,num=itmax,endpoint=False)
	indcell=np.arange(Ncell)
	
	for it in range(0,itmax):
		prob[it]=X
		xm[it]=np.sum((np.arange(Ncell)-n0)*np.abs(prob[it])**2)/np.sum(np.abs(prob[it])**2)
		X=np.matmul(U,X)

	ax=plt.gca()
	
	####
	data=np.load("tempdata/bspectrum.npz")
	qEs=data['qEs']
	beta=data['beta']
	data.close()
	ax.scatter(beta,qEs/F,c="red")
	####
	
	# ~ ax.set_ylim(-0.005,0.005)
	ax.set_xlim(0,2*np.pi)
	
	ax.scatter(F*time/h*4*np.pi,-xm,c="blue",label="Modèle tight-binding",s=1**2)
	ax.set_xlabel(r"$F*temps/h$")
	ax.set_ylabel("-<X>*F")
	plt.show()

	plt.savefig(wdir+"bloch.png", bbox_inches='tight',dpi=250)
	

	
	

