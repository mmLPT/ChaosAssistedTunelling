import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *

mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"Mi")
	os.mkdir(wdir+"Vi")

if mode=="compute":
	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	description=data['description']  
	e=data['e']
	h=data['h']
	gamma=data['gamma']
	data.close()
	
	# Initialization of potential and correcting the x0 value if needed
	pot=PotentialMP(e,gamma)
	
	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
	
	Ncell=nruns
	Npcell=64
	N=Npcell*Ncell
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, nruns=nruns,nruns2=int(0.5*(nruns+1)-1),Ncell=Ncell,N=N, e=e,gamma=gamma,h=h)
		
	# Initialization of the grid for given h value
	grid=Grid(N,h,xmax=Ncell*2*np.pi)

	# Create and diag the Floquet operator
	fo=CATFloquetOperator(grid,pot)
	
	i0=runid*Npcell
	
	M=np.zeros((N,Npcell),dtype=np.complex_)

	# Create a coherent state localized in x0 with width = 2.0 in x
	for i in range(0,Npcell):
		wf=WaveFunction(grid)
		wf.setState("diracx",i0=i0+i,norm=False) #Norm false to produce 'normalized' eigevalue
		fo.propagate(wf)
		wf.p2x()
		M[:,i]=wf.x
		
	# Save data
	np.savez(wdir+"Mi/"+str(runid),"w", M=M)

if mode=="gather":
	# This mode collect the spectrum for each value of h and make a single file

	# Reading inputfile
	data=np.load(wdir+"params.npz")
	nruns=data['nruns']
	N=int(data['N'])
	Ncell=int(data['Ncell'])
	h=data['h']
	data.close()

	# Create array to store data
	data=np.load(wdir+"Mi/"+str(0)+".npz")
	M=data['M']
	data.close()
	
	# For each runs read the output and add it to the new array
	for i in range(1,nruns):
		data=np.load(wdir+"Mi/"+str(i)+".npz")
		M=np.append(M,data['M'],axis=1)
		data.close()
		
	# Save the array
	np.savez(wdir+"gathered","w", M=M,N=N,Ncell=Ncell)

if mode=="compute2":
	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process
	
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	description=data['description']  
	e=data['e']
	h=data['h']
	gamma=data['gamma']
	data.close()

	data=np.load(wdir+"gathered.npz")
	N=int(data['N'])
	Ncell=int(data['Ncell'])
	M=data['M']
	data.close()
	
	# Initialization of potential and correcting the x0 value if needed

	
	nruns=int(sys.argv[4])+1 # Total number of // runs
	runid=int(sys.argv[5]) # Id of the current run
		
	pot=PotentialMP(e,gamma,idtmax=1000)
	grid=Grid(N,h,xmax=Ncell*2*np.pi)
	fo=CATFloquetOperator(grid,pot)
	
	x0=runid*2*np.pi
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)
	
	wf0=WaveFunction(grid)
	wf0.setState("coherent",x0=0.0,xratio=2.0)
	
	V=np.dot(wf0.x,M@wf.x)*grid.ddx
		
	# Save data
	np.savez(wdir+"Vi/"+str(runid),"w", V=V)
	
if mode=="gather2":
	data=np.load(wdir+"params.npz")
	nruns2=int(data['nruns2'])
	Ncell=int(data['Ncell'])
	data.close()

	# Create array to store data
	V=np.zeros(int(0.5*(Ncell-1)),dtype=complex)

	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns2):
		data=np.load(wdir+"Vi/"+str(i)+".npz")
		V[i]=data['V']
		data.close()
		
	# Save the array
	np.savez(wdir+"gathered2","w", V=V)
	
if mode=="plot":
	data=np.load(wdir+"gathered2.npz")
	V=data['V']
	data.close()

	ax=plt.gca()
	ax.set_ylim(10**(-30),10)
	ax.set_xlim(0,15)
	ax.set_yscale('log')
	plt.plot(np.arange(V.size),np.abs(V)**2,c='blue',zorder=0)
	plt.scatter(np.arange(V.size),np.abs(V)**2,c="red",s=5.0**2,zorder=1)
	print(np.abs(V)**2)
	ax.grid()
	plt.savefig(wdir+"V.png", bbox_inches='tight',dpi=250) 


