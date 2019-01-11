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

def imaginary(datafile="data/GS-avec-interactions-2",compute=True,read=True):
	if(compute):
		N=256*4
		gamma=0.3
		e=0.29
		h=0.3
		
		nuX=25.0
		nuL=8113.9
		nuperp=2.8*10**5
		
		a=5.*10**(-9)
		d=532*10**(-9)
		Nat=10**5

		g=np.pi*h**2*a*Nat*nuperp/nuL
		g=1.0
		
		gamma=0.0
		
		xmax=20*2*np.pi
		grid=Grid(N,h,xmax=xmax)
		
		omegax=(grid.h*250.0)/(2*8113.9)

		pot=PotentialMPasym(e,gamma,0,omegax)
		#pot=PotentialMPasym(e,gamma,0,(grid.h*25.0)/(2*8113.9))
		
		qitp=QuantumImaginaryTimePropagator(grid,pot,T0=4*np.pi,idtmax=1000,g=g)
		
		wf=WaveFunction(grid)
		wf.setState("diracp")
		wf.setState("coherent",xratio=25.0)
		wf=qitp.getGroundState(wf)
		wf.save(datafile)
		np.savez(datafile+"-dat","w", omegax=omegax, h=h, gamma=gamma, e=e,g=g)
		
	if(read):
		readw.wf(datafile)
		#~ data=np.load(datafile+".npz")
		#~ x=data["x"]
		#~ p=data["x"]
		#~ psix=data["psix"]
		#~ psip=data["psip"]
		
		#~ ax = plt.axes()
		#~ ax.set_xlim(min(x),max(x))
		#~ ax.set_ylim(0,1.1*max(abs(psix)**2))
		#~ plt.plot(x,abs(psix)**2,c="blue")
		
		#~ data2=np.load(datafile+"-dat.npz")
		#~ omega=data2['omegax']
		#~ h=data2['h']
		#~ gamma=data2['gamma']
		#~ e=data2['e']
		#~ g=data2['g']
	
		
		#~ #Rexp=3.0/(4.0*mxpsi)*((xmax/N)**0.5)**2
		#~ #gexp=2.0*omega**2*Rexp**3/3.0
		#~ Rth=(3*g1/(2*omega**2))**(1/3.0)
		
		#~ #phi=np.exp(-0.5*omega*x**2/h)*(omega/(np.pi*h))**0.25*(5*2*np.pi/N)**0.5
		#~ #phi=np.sqrt(3.0/(4.0*R))*np.sqrt(1-(x/R)**2)*(5*2*np.pi/N)**0.5
		
		#~ phi=np.zeros(N)
		#~ for i in range(0,N):
			#~ if abs(x[i])>Rth:
				#~ phi[i]=0
			#~ else:
				#~ phi[i]=np.sqrt(3.0/(4.0*R))*np.sqrt(1-(x[i]/R)**2)*(xmax/N)**0.5

		#~ plt.plot(x,phi**2,c="red")
		#~ plt.show()
