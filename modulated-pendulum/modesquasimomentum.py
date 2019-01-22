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

def free_prop_averaged(grid, pot,x0,compute=True,read=True,wdir="true_sim/averaged-test/"):
	if(compute):
		iperiod=40
		icheck=1
		n=int(iperiod/icheck)
		
		ibetamax=250

		xR=np.zeros(n)
		xL=np.zeros(n)
		time=np.zeros(n)
		
		for j in range(0,ibetamax):
			print(j)
			
			beta=np.random.normal(0.0, grid.h/6.0)
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
			
		np.savez(wdir+"data","w", time=time,ibetamax=ibetamax,n=n)
		
	if(read):
		data=np.load(wdir+"data.npz")
		time=data['time']
		ibetamax=data['ibetamax']
		n=data['n']

		xRav=np.zeros(n)
		xLav=np.zeros(n)
		
		for j in range(0,ibetamax):
			data=np.load(wdir+"run-"+strint(j)+".npz")
			xR=data['xR']
			xL=data['xL']
			for i in range(0,n):
				xRav[i]=xRav[i]+xR[i]
				xLav[i]=xLav[i]+xL[i]
				
		A=xRav[0]
		

		plt.plot(time,xLav/A, c="red")
		plt.plot(time,xRav/A, c="blue")
		
		ax=plt.gca()
		ax.set_xlabel(r"Périodes")
		ax.set_ylabel(r"$x gauche et x droite$")
		ax.set_title(r"$s=27.53 \ \nu=70.8 \ kHz \ \varepsilon=0.44 \ x_0=0.5 \pi$")
		ax.set_ylim(0,1.0)
		ax.set_xlim(0,70)
		
		plt.show()

		f= open(wdir+"simu.txt","w+")
		f.write("t gauche droite \n =============== \n")
		for i in range(0,n):
			f.write("{0:4.0f}\t {1:+7.5f}\t {2:+7.5f}\n".format(time[i],xLav[i]/A,xRav[i]/A))
			
		f.close()
