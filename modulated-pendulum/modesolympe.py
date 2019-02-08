import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
import modesbasic

def generate_run(wdir="input/",n=10):
        for i in range(0,10):
                np.savez(wdir+"input-"+str(i),"w", runid=i)

def run_info():
        data=np.load(sys.argv[1])
        wdir=sys.argv[2]+"/"
        return data['runid'], wdir

def distribution_omega(runid,grid,pot,Ndbeta=2.0,wdir="",ibetamax=ibetamax):
	# Compute the distribution of omega tunnel for a give distribution of
	# quasimomentum

	i=int(runid)
	
	beta=i*grid.h/ibetamax

	fo=CATFloquetOperator(grid,pot,beta=beta)
	
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=pot.x0,xratio=2.0)
	
	fo.diagonalize()
	fo.computeOverlapsAndQEs(wf)
	fo.getTunnelingPeriod()
	
	T=fo.getTunnelingPeriod()
	omega=2*np.pi/T

	np.savez(wdir+str(runid),"w", omega=omega,T=T,beta=beta)
			
def read_distribution_omega(n,wdir=""):
	if(read):
		omega=np.zeros(n)
		T=np.zeros(n)
		beta=np.zeros(n)
		for i in range(0,n):
			data=np.load(wdir+str(i)+".npz")
			omega[i]=data['omega']
			beta[i]=data['beta']
			T[i]=data['T']
			
		np.savez(wdir+"alldata","w", omega=omega,T=T,beta=beta)
	if(plot)
		data=np.load(wdir+"alldata.npz")
		beta=data['beta']
		omega=data['omega']
		plt.scatter(beta,omega)
		plt.show()
