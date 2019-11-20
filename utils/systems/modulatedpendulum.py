import numpy as np
import matplotlib.pyplot as plt
from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *
from utils.quantum.husimi import *
from utils.toolsbox import *

# This script contains:  4 classes and 4 functions

class PotentialMP(Potential):
	def __init__(self,e,gamma,f=np.cos,idtmax=1000):
		Potential.__init__(self)
		self.T0=4*np.pi
		self.idtmax=idtmax
		
		self.e=e
		self.gamma=gamma
		self.f=f # modulation waveform
		
		self.a1=getFourierCoefficient("a",1,self.f)
		self.b1=getFourierCoefficient("b",1,self.f)
		self.d1=(self.gamma-0.25)/(self.e*self.gamma)
		self.x0=self.R1()
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(1+self.e*self.f(t))*np.cos(x)
	
	def dVdx(self,x,t=np.pi/2.0):
		return self.gamma*(1+self.e*self.f(t))*np.sin(x)
		
	# The 4 following functions comes from classical analysis fo the bifurcation
	# they make possible to acess equilibrium positions for a given modulation waveform
	def thetaR1(self):
		if self.a1==0:
			v=0.25*np.pi*self.b1/abs(self.b1)
		elif self.a1>0.0:
			v=0.5*np.arctan(self.b1/self.a1)
		else:
			v=0.5*np.arctan(self.b1/self.a1)+0.5*np.pi
		return np.arctan2(np.sin(v)/2,np.cos(v))
		
	def thetaR2(self):
		if self.a1==0:
			v=0.25*np.pi*self.b1/abs(self.b1)
		elif self.a1>0.0:
			v=0.5*np.arctan(self.b1/self.a1)
		else:
			v=0.5*np.arctan(self.b1/self.a1)+0.5*np.pi
		return np.arctan2(np.sin(v+np.pi/2)/2,np.cos(v+np.pi/2))
		
	def R1(self):
		if self.d1>-0.5*np.sqrt(self.a1**2+self.b1**2):
			v=8.0/self.gamma*self.e*(0.5*np.sqrt(self.a1**2+self.b1**2)+self.d1)*self.gamma
		else:
			v=0.0
		return np.sqrt(v*(np.cos(self.thetaR1())**2+np.sin(self.thetaR1())**2/4))
		
	def R2(self):
		if self.d1>0.5*np.sqrt(self.a1**2+self.b1**2):
			v=8.0/self.gamma*self.e*(-0.5*np.sqrt(self.a1**2+self.b1**2)+self.d1)*self.gamma
		else:
			v=0.0
		return np.sqrt(v*(np.cos(self.thetaR2())**2+np.sin(self.thetaR2())**2/4))

		
class PotentialMPasym(PotentialMP):
	# Adding longitudinal confinment to modulated pendulum
	def __init__(self,e,gamma,x1,h):
		PotentialMP.__init__(self,e,gamma)
		self.x1=x1
		self.omega2= (getomegax(h))**2
		
	def Vx(self,x,t=np.pi/2.0):
		return PotentialMP.Vx(self,x,t)+self.Vxasym(x)
	
	def dVdx(self,x,t=np.pi/2.0):
		return PotentialMP.Vx(self,x,t)+self.omega2*(x-self.x1)
		
	def Vxasym(self,x):
		# Returns the non-symetric contribution of potential
		return 0.5*self.omega2*(x-self.x1)**2
		
	def braketVxasym(self,wf1,wf2):
		# Returns <wf1|Vxasym|wf2>
		return sum(np.conj(wf1.x)*self.Vxasym(wf1.grid.x)*wf2.x)
		
class PotentialMPGPE(PotentialMP):
	# Longitudinal confinment + GP
	def __init__(self,e,gamma,g,idtmax=1000):
		PotentialMP.__init__(self,e,gamma,idtmax=idtmax)
		self.isGP=True
		self.g=g
		
	def Vx(self,x,wfx,t=np.pi/2.0):
		return PotentialMP.Vx(self,x,t) +self.g*np.abs(wfx)**2

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

