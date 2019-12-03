import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from scipy.linalg import expm, sinm, cosm
from utils.mathtools.periodicfunctions import *


# State: stable [22/02/2019]

# To be used with "run-spectrum.slurm"

# This scripts makes possibles to 
# 1. compute in // the spectrum of qE for differents value of [h]
# 2. gather the information
# 3. plot the spectrum

# Arguments to provide:
# 1. mode = "compute", "gather", "plot"
# 2. working directory
# if mode=="compute":
# 	3. input file
# 	4. total number of tasks
# 	5. id of the current runs

mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"runs")

if mode=="compute":
	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process
	

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	description=data['description']  

	N=int(data['N'])
	e=data['e']
	gamma=data['gamma']
	x0=data['x0'] 
	h=data['h']

	data.close()

	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, nruns=nruns,N=N, e=e,gamma=gamma,x0=x0,h=h)

	# Creating array to store data
	ins=np.zeros(N)
	qEs=np.zeros(N)
	overlaps=np.zeros(N,dtype=complex)
	symX=np.zeros(N,dtype=bool)

	# Create and diag the Floquet operator
	grid=Grid(N,h)
	beta=np.linspace(-0.5,0.5,nruns)[runid]*h
	mod=PeriodicFunctions()
	pot=PotentialMP(e,gamma)
	fo=CATFloquetOperator(grid,pot,beta=beta)
	fo.diagonalize()

	# Create a coherent state localized in x0 with width = 2.0 in x
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=x0,xratio=2.0)

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	ind, overlaps=fo.getOrderedOverlapsWith(wfcs)
	ind=np.flipud(ind)
	overlaps=np.flipud(overlaps)
	for i in range(0,N):
		qEs[i]=fo.getQE(ind[i])
		symX[i]=fo.getEvec(ind[i]).isSymetricInX()
		
	# Save data
	np.savez(wdir+"runs/"+str(runid),"w", h=h, qEs=qEs, overlaps=overlaps,symX=symX,beta=beta)

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
	h=np.zeros((nruns,N))
	beta=np.zeros((nruns,N))
	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns):
		data=np.load(wdir+"runs/"+str(i)+".npz")
		qEs[i]=data['qEs']
		beta[i]=data['beta']
		overlaps[i]=data['overlaps']
		symX[i]=data['symX']
		h[i]=data['h']
		data.close()

	# Save the array
	np.savez(wdir+"gathered","w", e=e,gamma=gamma,h=h,qEs=qEs,overlaps=overlaps,symX=symX,nruns=nruns,beta=beta)

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
	data.close()
	
	beta=beta[:,-1]/h[:,-1]
	qEs=qEs[:,-1]
	
	beta=beta*2*np.pi
	dbeta=np.abs(beta[0]-beta[1])
	
	print(dbeta)
	
	nfft=nruns/2
	V=np.fft.rfft(qEs)/nruns
	# ~ V=np.fft.fftshift(V)
	X=np.arange(V.size)
	
	vint=np.sqrt(np.trapz((np.gradient(qEs,beta))**2,dx=dbeta)/(h[0,0]**2*2*np.pi))
	vcoupling=np.sqrt(2*np.sum(X**2*np.abs(V)**2))/h[0,0]
	print("1/h=",1/h[0,-1])
	print("vint={:.2E}".format(float(vint)))
	print("vcoupling={:.2E}".format(float(vcoupling)))
	print(vint/vcoupling)

	# General setup for plotting
	ax=plt.gca()
	ax.set_xlabel(r"$\beta$")
	ax.set_ylabel(r"$qE/h$")
	ax.set_xlim(-np.pi,np.pi)
	emin=np.min(qEs)
	emax=np.max(qEs)
	dE=0.2*(emax-emin)
	ax.set_ylim(emin-dE,emax+dE)
	# ~ print(2*np.min(qEs),2*np.max(qEs))

	ax.plot(beta,qEs,c="tab:blue")
	# ~ ax.plot(beta,np.fft.irfft(V)*nruns,c="tab:red")
	
	data=np.load("tempdata/states.npz")
	beta0=data['beta']
	qEs0=data['qEs']
	data.close()
	
	V0=np.fft.rfft(qEs0)
	print(np.abs(V0)**2)
	# ~ print(np.abs(V)**2)

	# ~ print(np.fft.irfft(np.fft.rfft(qEs0), 2*len(V0)-1))
	
	ax.plot(beta0,np.fft.irfft(np.fft.rfft(qEs0), 2*len(V0)-1),c="tab:red")
	ax.scatter(beta0,qEs0,c="tab:red",zorder=4)
	
	
	
	
	plt.savefig(wdir+"spectrum.png", bbox_inches='tight',dpi=250) 
	
	ax.clear()
	ax=plt.gca()
	ax.set_xlabel(r"$n$")
	ax.set_ylabel(r"$Vn$")
	ax.set_xlim(0,15)
	ax.set_yscale("log")
	# ~ ax.set_ylim(10**(-17),10**(-7))

	plt.scatter(X,np.abs(V/h[0,0])**2,c="blue")
	plt.plot(X,np.abs(V/h[0,0])**2,c="blue")
	
	plt.scatter(X,X**2*np.abs(V/h[0,0])**2,c="red")
	plt.plot(X,X**2*np.abs(V/h[0,0])**2,c="red")

	plt.savefig(wdir+"V.png", bbox_inches='tight',dpi=250)
	
	
	
	#changer ce h[0,0] qui n'a aucun sens
	
	
	# ~ Nc=36*10
	# ~ M=np.zeros((Nc,Nc),dtype=np.complex_)
	# ~ for i in range(0,Nc):
		# ~ for j in range(0,Nc):
			# ~ l=int(min(np.abs(i-j),np.abs(Nc+i-j)))
			# ~ if i>j:
				# ~ M[i,j]=V2[l]
			# ~ else:
				# ~ M[i,j]=np.conjugate(V2[l])
		
	
		
	
	
	# ~ ax.set_xlabel(r"$i$")
	# ~ ax.set_ylabel(r"$|\langle n | \mathcal{H} | n+1 \rangle|^2$")
	
	# ~ ax.set_xlim(0,n2)
	
	# ~ mx=np.arange(n2)
	# ~ plt.scatter(mx,(np.abs(V))**2)
	
	# ~ plt.plot(np.arange(V2.size),np.abs(V2)**2,c='blue',zorder=0)
	# ~ plt.scatter(np.arange(V2.size),np.abs(V2)**2,c="red",s=5.0**2,zorder=1)
	
	# ~ U0=expm(-1j*M)
	# ~ for j in range(0,5):
		# ~ print(1+5*j)
		# ~ U=np.linalg.matrix_power(U0,1+5*j)
		# ~ V3=np.zeros(Nc,dtype=complex)	
		# ~ for i in range(0,Nc):
			# ~ V3[i]=U[0,i]
		# ~ plt.plot(np.arange(V3.size),np.abs(V3)**2,zorder=0)
		# ~ plt.scatter(np.arange(V3.size),np.abs(V3)**2,s=5.0**2,zorder=1)
	
	# ~ ax.set_yscale('log')
	
	# ~ ax.set_ylim(10**(-8),1)
	# ~ ax.set_xlim(0,25)

	
	# ~ ax.grid()		
				
	# ~ plt.savefig(wdir+"V.png", bbox_inches='tight',dpi=250) 
	

