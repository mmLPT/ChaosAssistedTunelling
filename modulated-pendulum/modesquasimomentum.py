import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
#from utils.mathtools.periodicfunctions import *
import utils.plot.read as readw
import modesbasic

def free_prop_averaged(grid, pot,x0,Ndbeta=2.0,ibetamax=1,iperiod=100,compute=True,read=True,wdir="averaged-beta/noname/"):
	# Mode to study the free propagation with split step method
	# We look at "left" and "right" observable over time which micmic 
	# the exprimental setup.
	# The observables are incoheretly 
	if(compute):
		iperiod=iperiod
		icheck=1
		n=int(iperiod/icheck)
		
		ibetamax=ibetamax

		xR=np.zeros(n)
		xL=np.zeros(n)
		time=np.zeros(n)
		
		for j in range(0,ibetamax):
			print(j,"/",ibetamax-1)
			
			
			beta=np.random.normal(0.0, grid.h/(3.0*Ndbeta))
			if ibetamax==1:
				beta=0.0
				
			fo=CATFloquetOperator(grid,pot,beta=beta)
			
			wf=WaveFunction(grid)
			wf.setState("coherent",x0=x0,xratio=2.0)
			
			for i in range(0,iperiod):
				if i%icheck==0:
					xL[int(i/icheck)]=wf.getxL()
					xR[int(i/icheck)]=wf.getxR()
					time[int(i/icheck)]=i*2
				fo.propagate(wf)
			
			np.savez(wdir+"run-"+strint(j),"w", beta=beta, xL = xL, xR=xR)
			
		np.savez(wdir+"data","w", time=time,ibetamax=ibetamax,n=n,h=grid.h,paramspot=pot.getParams(),x0=x0)
		
	if(read):
		data=np.load(wdir+"data.npz")
		time=data['time']
		ibetamax=data['ibetamax']
		n=data['n']
		e=data['paramspot'][0]
		gamma=data['paramspot'][1]
		x0=data['x0']
		h=data['h']
		s,nu,x0exp=modesbasic.convert2exp(gamma,h,x0)

		xRav=np.zeros(n)
		xLav=np.zeros(n)
		
		for j in range(0,ibetamax):
			data=np.load(wdir+"run-"+strint(j)+".npz")
			xR=data['xR']
			xL=data['xL']
			plt.plot(time/2.0,xL, c="red")
			plt.plot(time/2.0,xR, c="blue")
			for i in range(0,n):
				xRav[i]=xRav[i]+xR[i]
				xLav[i]=xLav[i]+xL[i]
				
		plt.show()
				
		A=xRav[0]
		

		plt.plot(time/2.0,xLav/A, c="red")
		plt.plot(time/2.0,xRav/A, c="blue")
		
		#~ x1,y1=np.loadtxt("exp-data/pop_non_tunnel.txt",usecols=(0, 1), unpack=True)
		#~ x2,y2=np.loadtxt("exp-data/pop_tunnel.txt",usecols=(0, 1), unpack=True)
		
		#~ plt.scatter(x1,y1)
		#~ plt.scatter(x2,y2)
		
		
		ax=plt.gca()
		ax.set_xlabel(r"Périodes")
		ax.set_ylabel(r"$Observable de position normalisée$")
		ax.set_title(r"$s={:.2f} \quad \nu={:.2f}\ kHz \quad \varepsilon={:.2f}  \quad x_0={:.0f}^\circ$".format(s,nu*10**(-3),e,x0exp))
		ax.set_ylim(0,1.0)
		ax.set_xlim(0,max(time/2.0))
		
		plt.show()

		#~ f= open(wdir+"simu.txt","w+")
		#~ f.write("t gauche droite \n =============== \n")
		#~ for i in range(0,n):
			#~ f.write("{0:4.0f}\t {1:+7.5f}\t {2:+7.5f}\n".format(time[i],xLav[i]/A,xRav[i]/A))
			
		#~ f.close()
		
def distribution_omega(grid,pot,Ndbeta=2.0,compute=True,read=True,ibetamax=1500,wdir="CAT/",datafile="distribution",scan=False):
	# Compute the distribution of omega tunnel for a give distribution of
	# quasimomentum
	
	omega=np.zeros(ibetamax)
	T=np.zeros(ibetamax)
	beta=np.zeros(ibetamax)
	
	if(compute):
		for i in range(0,ibetamax):
			if(scan):
				beta[i]=i*grid.h/ibetamax
			else:
				beta[i]=np.random.normal(0.0, grid.h/(3.0*Ndbeta))
			
			fo=CATFloquetOperator(grid,pot,beta=beta[i])
			
			wf=WaveFunction(grid)
			wf.setState("coherent",x0=pot.x0,xratio=2.0)
			
			fo.diagonalize()
			fo.computeOverlapsAndQEs(wf)
			fo.getTunnelingPeriod()
			
			T[i]=fo.getTunnelingPeriod()
			omega[i]=2*np.pi/T[i]
			
			print(i,omega[i],T[i])

		np.savez(wdir+datafile,"w", omega=omega,T=T,beta=beta)
			
	if(read):
		data=np.load(wdir+datafile+".npz")
		omega=data['omega']
		beta=data['beta']
		#plt.hist(omega, 15, normed=True)
		plt.scatter(beta,omega)
		plt.show()
