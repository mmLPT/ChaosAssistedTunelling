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
	nruns=data['nruns']
	symX=data['symX']
	data.close()

	# General setup for plotting
	ax=plt.gca()
	ax.set_xlabel(r"$\beta$")
	ax.set_ylabel(r"$qE/h$")
	# ~ ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f}$".format(e,gamma))
	ax.set_xlim(-0.5,0.5)

	cmap = plt.cm.get_cmap('Reds')

	overlaps=np.abs(overlaps)

	# ~ qEs=4*np.pi*(qEs/h)/np.pi
	

	ind=np.argmax(overlaps,axis=1)

	n=nruns
	beta2=np.zeros(n)
	qEs2=np.zeros(n)
	for i in range(0,n):
		j=np.argmax(overlaps[i])
		beta2[i]=beta[i,j]/h[0,0]
		qEs2[i]=qEs[i,j]
		
	

	
	plt.scatter(beta2,qEs2)
	plt.savefig(wdir+"spectrum.png", bbox_inches='tight',dpi=250) 
	
	ax.clear()
	

	
	n2=n
	# ~ V=np.zeros(n2,dtype=complex)
	# ~ for i in range(0,n2):
		# ~ V[i]=np.sum(np.exp(-1j*2*np.pi*beta2*i)*qEs2)/n*4*np.pi/h[0,0]
		
	V2=np.fft.fft(qEs2)/n*4*np.pi/h[0,0]
	
	
	Nc=36*10
	M=np.zeros((Nc,Nc),dtype=np.complex_)
	for i in range(0,Nc):
		for j in range(0,Nc):
			l=int(min(np.abs(i-j),np.abs(Nc+i-j)))
			if i>j:
				M[i,j]=V2[l]
			else:
				M[i,j]=np.conjugate(V2[l])
			# ~ M[i,j]=np.exp(-3*l)
			
	# ~ U=expm(-1j*M)
	# ~ U=np.linalg.matrix_power(U,5)
	# ~ V3=np.zeros(Nc,dtype=complex)	
	# ~ for i in range(0,Nc):
		# ~ V3[i]=U[0,i]
		
	
		
	
	
	ax.set_xlabel(r"$i$")
	ax.set_ylabel(r"$|\langle n | \mathcal{H} | n+1 \rangle|^2$")
	
	# ~ ax.set_xlim(0,n2)
	
	mx=np.arange(n2)
	# ~ plt.scatter(mx,(np.abs(V))**2)
	
	# ~ plt.plot(np.arange(V2.size),np.abs(V2)**2,c='blue',zorder=0)
	# ~ plt.scatter(np.arange(V2.size),np.abs(V2)**2,c="red",s=5.0**2,zorder=1)
	
	U0=expm(-1j*M)
	for j in range(0,5):
		print(1+5*j)
		U=np.linalg.matrix_power(U0,1+5*j)
		V3=np.zeros(Nc,dtype=complex)	
		for i in range(0,Nc):
			V3[i]=U[0,i]
		plt.plot(np.arange(V3.size),np.abs(V3)**2,zorder=0)
		plt.scatter(np.arange(V3.size),np.abs(V3)**2,s=5.0**2,zorder=1)
	
	ax.set_yscale('log')
	
	ax.set_ylim(10**(-8),1)
	ax.set_xlim(0,25)
	# ~ for i in range(0,15):
		

	
	ax.grid()		
				
	plt.savefig(wdir+"V.png", bbox_inches='tight',dpi=250) 
	

