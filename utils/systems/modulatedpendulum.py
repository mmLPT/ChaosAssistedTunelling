import numpy as np
import matplotlib.pyplot as plt
from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *
from utils.quantum.husimi import *

# This script contains:  4 classes and 4 functions

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
		
	# The 4 following functionseee
		
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
	# Adding longitudinal confinment to modulated pendulum
	def __init__(self,e,gamma,x1,omega):
		PotentialMP.__init__(self,e,gamma)
		self.x1=x1
		self.omega2= omega**2
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(1+self.e*self.f(t))*np.cos(x)+0.5*self.omega2*(x-self.x1)**2
	
	def dVdx(self,x,t=0):
		return self.gamma*(1+self.e*self.f(t))*np.sin(x)+self.omega2*x*(x-self.x1)
		
class PotentialTest(Potential):
	# Adding longitudinal confinment to modulated pendulum
	def __init__(self):
		Potential.__init__(self)
		self.omega=2.0
		self.x1=0.0
		
	def Vx(self,x):
		return 0.5*self.omega**2*(x-self.x1)**2

class H0(QuantumOperator):
	# Hamiltonian for unmodulated pendulum. The matric p representation
	# is build from analytical results
	
	def __init__(self,grid, gamma):
		QuantumOperator.__init__(self,grid)
		self.gamma=gamma
		
	def fillM(self):
		# Matrix p representation is built from analytical results
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
		# Get the ground state
		wf=WaveFunction(self.grid)
		wf.setState("loadp",psip=self.eigenvec[0].p*np.exp(-(1j/self.grid.h)*x0*self.grid.p))
		return wf
		
class QuantumImaginaryTimePropagator(QuantumOperator):
	def __init__(self,grid,potential,hermitian=False,idtmax=1,T0=1):
		self.potential=potential
		self.grid=grid
		self.dt=T0/idtmax
		a=5.0#e-9
		m=86.9091835*1.660538921 #e-27
		hbar=6.62607015/(2*np.pi)  #e-34
		nuL=8113.9
		self.g=self.grid.h**2*(hbar*a/(m*nuL)*1.0e-16)
		self.g=15000.0
		self.Up=np.zeros(grid.N,dtype=np.complex_)
		self.Up=np.exp(-(self.dt/self.grid.h)*(grid.p**2/2))
		
	def getGroundState(self,wf):
		mudiff=1.0
		wf0x=wf.x
		mu0=1.0


		while mudiff > 1.0e-8:
			wf.x=wf.x*np.exp(-(self.dt/self.grid.h)*(self.potential.Vx(self.grid.x)+self.g*abs(wf.x)**2)/2.0) #+self.g*abs(wf.x)**2
			wf.x2p()
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*np.exp(-(self.dt/self.grid.h)*(self.potential.Vx(self.grid.x)+self.g*abs(wf.x)**2)/2.0)
			
			mu=(self.grid.h/self.dt)*np.log(abs(wf0x[int(self.grid.N/2.0)]/wf.x[int(self.grid.N/2.0)]))
			
			wf.normalizeX()
		
			mudiff=abs(mu-mu0)
			wf0x=wf.x
			mu0=mu
			print("norm =",abs(wf%wf),"mudiff=",mudiff/(1.0e-8))
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

# READY TO USE FUNCITONS DIRTY BUT CONVENIENT
		
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
	# Regarde comment evolue la periode
	npoints=11
	x1=np.linspace(-10*2*np.pi,10*2*np.pi,npoints)
	
	asym=np.zeros(npoints)
	T=np.zeros(npoints)
	Tth=np.zeros(npoints)
	qE=np.zeros(npoints)
	
	husimi=Husimi(grid)
	pot=PotentialMPasym(e,gamma,0,(grid.h*25.0)/(2*8113.9))
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	
	fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
	fo.diagonalize()
	fo.findTunellingStates(wfcs)
	T0,qE0=fo.getSplitting()

	for i in range(0,npoints):
		
		pot=PotentialMPasym(e,gamma,x1[i],(grid.h*25.0)/(2*8113.9))
		asym[i]=abs(pot.Vx(-pot.x0)-pot.Vx(pot.x0))/(4*np.pi)
		
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i],qE[i]=fo.getSplitting()
		Tth[i]=T0/np.sqrt(1+(asym[i]/qE0)**2)
		#fo.saveTunellingStates(husimi,wdir+"evec/"+strint(i))

		print("T=",T[i],"Tth=",Tth[i],"asym=", asym[i])
	np.savez(wdir+datafile,"w", x1=x1, asym=asym, T=T, qE=qE, Tth=Tth)

def explore_N_impact(e,gamma):
	# Pour voir l impact de N sur le taux tunnel
	# conclusion : si x=[A,A[ ok si x=[A,A] ca deconne
	pot=PotentialMP(e,gamma)
	
	h=0.1
	for N in [64,128,256,512,1024,2048]:
		grid=Grid(N,h)
		
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=-pot.x0,xratio=2.0)
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T,qE=fo.getSplitting()
		
		print("T=",T,"N=",N)

