import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from scipy.interpolate import interp1d

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.sawtooth import *

import scipy.special as sc



# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data
 
mode=sys.argv[1]
wdir=sys.argv[2]

nalpha=71

if mode=="initialize":
	inputfile=sys.argv[3]
	
	data=np.load(inputfile+".npz")
	Ntable=data['Ntable']
	data.close()
	
	alpha=np.linspace(0.0,1.0,nalpha)
	
	os.mkdir(wdir)
	os.mkdir(wdir+"raw-data")
	for i in range(0,alpha.size):
		os.mkdir(wdir+"raw-data/a-"+str(i))
		for j in Ntable:
			os.mkdir(wdir+"raw-data/a-"+str(i)+"/N-"+str(j))
		
	os.mkdir(wdir+"pictures")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	
	nruns=int(sys.argv[4]) # number of runs for a given h
	runid=int(sys.argv[5])-1 # Id of the current run
	
	data=np.load(inputfile+".npz")
	Ntable=data['Ntable']
	nrunsN=data['nrunsN']
	Ninfo=data['Ninfo']
	potential=data['potential']
	#Ninfo[0,i] : N value of runs i
	#Ninfo[1,i] : relative index of runs i ('i-th over current N')
	q=data['q']
	beta=data['beta']
	data.close()
	
	
	# ~ alpha=np.linspace(0.0,1.0,nalpha)
	alpha=np.linspace(0.0,1.0,nalpha)
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",alpha=alpha,nruns=nruns,Ninfo=Ninfo,Ntable=Ntable,nrunsN=nrunsN,beta=beta,potential=potential)

	for ia in range(0,alpha.size):
		if potential=="RS":
			pot=PotentialST(alpha[ia])
			grid=Grid(Ninfo[0,runid],h=2*np.pi,xmax=2*np.pi)
		if potential=="GG":
			pot=PotentialGG(alpha[ia]*np.pi,np.pi)
			grid=Grid(Ninfo[0,runid],h=2*np.pi,xmax=2*np.pi)
		
		fo=CATFloquetOperator(grid,pot,randomphase=True)
		fo.diagonalize()
	
		momenta=0
		for iN in range(0,Ninfo[0,runid]):
			momenta+=fo.getEvec(iN).getMomentum("p",2)
		momenta/=Ninfo[0,runid]
		np.savez(wdir+"raw-data/a-"+str(ia)+"/N-"+str(Ninfo[0,runid])+"/"+str(Ninfo[1,runid]),momenta=momenta)

if mode=="average":
	data=np.load(wdir+"params.npz")
	Ntable=data['Ntable']
	nrunsN=data['nrunsN']
	alpha=data['alpha']
	data.close()
	
	runid=int(sys.argv[3])-1
	
	momenta=np.zeros(Ntable.size)
	for i in range(0,Ntable.size):
		momt=np.zeros(nrunsN[i])
		for j in range(0,nrunsN[i]):
			data=np.load(wdir+"raw-data/a-"+str(runid)+"/N-"+str(Ntable[i])+"/"+str(j)+".npz")
			momt[j]=data['momenta']
			data.close()
		momenta[i]=np.mean(momt)
		
		
	fit = np.polyfit(np.log(Ntable),np.log(momenta), 1)
	D2=-fit[0]
		
	ax=plt.gca()
	ax.set_title(r"$q={:.2f}$".format(alpha[runid]))
		
	plt.scatter(np.log(Ntable),np.log(momenta))
	plt.plot(np.log(Ntable),fit[0]*np.log(Ntable)+fit[1],c="red")
	plt.savefig(wdir+"pictures/"+strint(runid)+".png", bbox_inches = 'tight',format="png")
	
	np.savez(wdir+"raw-data/a-"+str(runid)+"/d2",D2=D2)
	
if mode=="gather":
	data=np.load(wdir+"params.npz")
	alpha=data['alpha']
	data.close()
	
	D2=np.zeros(alpha.size)
	
	for i in range(0,alpha.size):
		data=np.load(wdir+"raw-data/a-"+str(i)+"/d2.npz")
		D2[i]=data['D2']
		data.close()
		
	np.savez(wdir+"d2-final",alpha=alpha,D2=D2)
		
if mode=="plot":
	data=np.load(wdir+"d2-final.npz")
	alpha=data['alpha']
	D2=data['D2']
	data.close()
	
	ax=plt.gca()
	ax.set_xlim(np.min(alpha),np.max(alpha))
	ax.set_ylim(0.0,1.0)
	ax.set_xlabel(r"a")
	ax.set_ylabel(r"$D_2$")
	
	D2th=(2*alpha*sc.gamma(2-0.5*np.ones(alpha.size)))/(np.sqrt(np.pi)*sc.gamma(2))
	
	ax.grid()
	
	plt.scatter(alpha,D2,c="red")
	plt.plot(alpha,D2th,c="blue")

	
	plt.savefig(wdir+"d2.png", bbox_inches = 'tight',format="png")



				
	


