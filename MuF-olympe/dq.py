import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.plot.latex import *
from utils.systems.modulatedpendulum import *
from utils.systems.sawtooth import *



# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data
 
mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	inputfile=sys.argv[3]
	
	data=np.load(inputfile+".npz")
	Ntable=data['Ntable']
	q=data['q']
	data.close()
	
	os.mkdir(wdir)
	os.mkdir(wdir+"raw-data")
	for i in range(0,q.size):
		os.mkdir(wdir+"raw-data/q-"+str(i))
		for j in Ntable:
			os.mkdir(wdir+"raw-data/q-"+str(i)+"/N-"+str(j))
		
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
	#Ninfo[0,i] : N value of runs i
	#Ninfo[1,i] : relative index of runs i ('i-th over current N')
	q=data['q']
	alpha=data['alpha']
	data.close()
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",alpha=alpha,nruns=nruns,q=q,Ninfo=Ninfo,Ntable=Ntable,nrunsN=nrunsN)

	pot=PotentialST(alpha)
	grid=Grid(Ninfo[0,runid],h=2*np.pi,xmax=2*np.pi)
	fo=CATFloquetOperator(grid,pot,randomphase=True)
	fo.diagonalize()

	for iq in range(0,q.size):
		momenta=0
		for iN in range(0,Ninfo[0,runid]):
			momenta+=fo.getEvec(iN).getMomentum("p",q[iq])
		momenta/=Ninfo[0,runid]
		np.savez(wdir+"raw-data/q-"+str(iq)+"/N-"+str(Ninfo[0,runid])+"/"+str(Ninfo[1,runid]),momenta=momenta)

if mode=="average":
	data=np.load(wdir+"params.npz")
	Ntable=data['Ntable']
	nrunsN=data['nrunsN']
	q=data['q']
	alpha=data['alpha']
	data.close()
	
	runid=int(sys.argv[3])-1
	
	momenta=np.zeros(Ntable.size)
	for i in range(0,Ntable.size):
		momt=np.zeros(nrunsN[i])
		for j in range(0,nrunsN[i]):
			data=np.load(wdir+"raw-data/q-"+str(runid)+"/N-"+str(Ntable[i])+"/"+str(j)+".npz")
			momt[j]=data['momenta']
			data.close()
		momenta[i]=np.median(momt)
		
		
	fit = np.polyfit(np.log(Ntable),np.log(momenta), 1)
	if not(q[runid]==1):
		Dq=-fit[0]/(q[runid]-1)
	else:
		Dq=3.0
		
	ax=plt.gca()
	ax.set_title(r"$q={:.2f}$".format(q[runid]))
		
	plt.scatter(np.log(Ntable),np.log(momenta))
	plt.plot(np.log(Ntable),fit[0]*np.log(Ntable)+fit[1],c="red")
	plt.savefig(wdir+"pictures/"+strint(runid)+".png", bbox_inches = 'tight',format="png")
	
	np.savez(wdir+"raw-data/q-"+str(runid)+"/dq",Dq=Dq)
	
if mode=="gather":
	data=np.load(wdir+"params.npz")
	q=data['q']
	alpha=data['alpha']
	data.close()
	
	Dq=np.zeros(q.size)
	
	for i in range(0,q.size):
		data=np.load(wdir+"raw-data/q-"+str(i)+"/dq.npz")
		Dq[i]=data['Dq']
		data.close()
		
	np.savez(wdir+"dq-final",q=q,alpha=alpha,Dq=Dq)
		
if mode=="plot":
	data=np.load(wdir+"dq-final.npz")
	q=data['q']
	alpha=data['alpha']
	Dq=data['Dq']
	data.close()
	
	ax=plt.gca()
	ax.set_xlim(-4,4)
	ax.set_ylim(0.0,2.0)
	ax.set_title(r"$\alpha={:.2f}$".format(alpha))
	ax.set_xlabel(r"q")
	ax.set_ylabel(r"D_q")
	
	plt.scatter(q,Dq,c="red")
	plt.savefig(wdir+"dq.png", bbox_inches = 'tight',format="png")

				
	


