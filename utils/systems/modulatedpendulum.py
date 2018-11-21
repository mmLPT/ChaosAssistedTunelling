import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *
from utils.quantum.husimi import *


class PotentialMP(Potential):
	def __init__(self,e,gamma,f=np.cos):
		Potential.__init__(self)
		self.e=e
		self.gamma=gamma
		self.f=f #np.cos #PeriodicFunctions().triangle
		self.a1=getFourierCoefficient("a",1,self.f)
		self.b1=getFourierCoefficient("b",1,self.f)
		self.d1=(self.gamma-0.25)/(self.e*self.gamma)
		self.x0=self.R1()
		self.isTimeDependent=True
		
	def Vx(self,x,t=0):
		return -self.gamma*(1+self.e*self.f(t))*np.cos(x)
	
	def dVdx(self,x,t=0):
		return self.gamma*(1+self.e*self.f(t))*np.sin(x)
		
	def R1(self):
		if self.d1>-0.5*np.sqrt(self.a1**2+self.b1**2):
			v=8.0/self.gamma*self.e*(0.5*np.sqrt(self.a1**2+self.b1**2)+self.d1)*self.gamma
		else:
			v=0.0
		return np.sqrt(v)
		
	def R2(self):
		if self.d1>0.5*np.sqrt(self.a1**2+self.b1**2):
			v=8.0/self.gamma*self.e*(-0.5*np.sqrt(self.a1**2+self.b1**2)+self.d1)*self.gamma
		else:
			v=0.0
		return np.sqrt(v)
	
	def thetaR1(self):
		if self.a1==0:
			v=0.25*np.pi*self.b1/abs(self.b1)
		elif self.a1>0.0:
			v=0.5*np.arctan(self.b1/self.a1)
		else:
			v=0.5*np.arctan(self.b1/self.a1)+0.5*np.pi
		return v
		
	def thetaR2(self):
		return self.thetaR1()+np.pi/2.0
		
class PotentialMPasym(PotentialMP):
	def __init__(self,e,gamma,x3,h):
		PotentialMP.__init__(self,e,gamma)
		self.x3=x3
		self.omega1=h**2*(25.0/8113.9)**2/8.0
		
	def Vx(self,x,t=0):
		return -self.gamma*(1+self.e*self.f(t))*np.cos(x)+0.5*self.omega1*(x-self.x3)**2
	
	def dVdx(self,x,t=0):
		return self.gamma*(1+self.e*self.f(t))*np.sin(x)+self.omega1*x*(x-self.x1)

class H0(QuantumOperator):
	def __init__(self,grid, gamma):
		QuantumOperator.__init__(self,grid)
		self.gamma=gamma
		
	def fillM(self):
		self.Mrepresentation="p"
		self.hermitian=True
		p=np.fft.fftshift(self.grid.p)
		for li in range(0,self.N):
			for co in range(0,self.N):
				if li==co:
					self.M[li][co]=p[li]**2/2
				elif (li==co+1)|(li==co-1):
					self.M[li][co]=-self.gamma/(2*2*self.h*np.pi)
				else:
					self.M[li][co]=0
					
	def getGS(self, x0):
		wf=WaveFunction(self.grid)
		wf.setState("loadp",psip=self.eigenvec[0].p*np.exp(-(1j/self.grid.h)*x0*self.grid.p))
		return wf
"""			
class StrobosopicPhaseSpaceMP(StrobosopicPhaseSpace):
	def sety0(self):
		self.y0=np.array([rd.randint(0,101)/100.0*2*np.pi-np.pi,rd.randint(0,101)/100.0*2*1.3-1.3])
		
	def npz2plt(self, potential, datadir=""):
		R1=potential.R1()
		R2=potential.R2()
		thetaR1=potential.thetaR1()
		thetaR2=potential.thetaR2()
		plt.scatter(R1*np.cos(thetaR1),0.5*R1*np.sin(thetaR1),s=3**2,marker="o",c="red")
		plt.scatter(R2*np.cos(thetaR2),0.5*R2*np.sin(thetaR2),s=3**2,marker="o",c="red")
		plt.scatter(R1*np.cos(thetaR1+np.pi),0.5*R1*np.sin(thetaR1+np.pi),s=3**2,marker="o",c="red")
		plt.scatter(R2*np.cos(thetaR2+np.pi),0.5*R2*np.sin(thetaR2+np.pi),s=3**2,marker="o",c="red")
		StrobosopicPhaseSpace.npz2plt(self,datadir)
"""		
# READY TO USE FUNCITONS
		
def splitting_with_h( N, h, e, gamma, asym=False, xasym=10*np.pi, datafile="split"):
	imax=h.shape[0]
	T=np.zeros(imax)
	qE=np.zeros(imax)

	for i in range(0,imax):
		print(str(i+1)+"/"+str(imax)+" - h="+str(h[i]))
		grid=Grid(N,h[i],2*np.pi)
		if asym==True:
			pot=PotentialMPasym(e,gamma,xasym,h[i])
		else:
			pot=PotentialMP(e,gamma)
			
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
		
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i],qE[i]=fo.getSplitting()
		
	np.savez(datafile,"w", h=h, T=T,qE=qE)

def explore_e_gamma(N):
	et=np.linspace(0.3,0.35,10)
	gammat=np.linspace(0.27,0.30,6)
	ht=np.linspace(np.sqrt(4*0.27/30),np.sqrt(4*0.30/5),200)
	for e in et:
		for gamma in gammat:
			print("** e="+str(e)+" -- gamma="+str(gamma)+" **")
			datafile="explore_e_gamma/split__e_"+str(e)+"__gamma_"+str(gamma)
			splitting_with_h( N, ht, e, gamma, datafile=datafile)
			
def explore_asymetry(grid,e,gamma,wdir="asym/",datafile="data"):

	npoints=11
	x1=np.linspace(-10*2*np.pi,10*2*np.pi,npoints)
	
	asym=np.zeros(npoints)
	T=np.zeros(npoints)
	qE=np.zeros(npoints)
	
	husimi=Husimi(grid)
	pot=PotentialMP(e,gamma)
	
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
	fo.diagonalize()
	fo.findTunellingStates(wfcs)
	T0,qE0=fo.getSplitting()

	for i in range(0,npoints):
		pot=PotentialMPasym(e,gamma,x1[i],grid.h)
		asym[i]=abs(pot.Vx(-pot.x0)-pot.Vx(pot.x0))/(4*np.pi)
		
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=2500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i],qE[i]=fo.getSplitting()
		#fo.saveTunellingStates(husimi,wdir+"evec/"+strint(i))

		print(T[i],T0/np.sqrt(1+(asym[i]/qE0)**2), asym[i])
	np.savez(wdir+datafile,"w", x1=x1, asym=asym, T=T, qE=qE)
