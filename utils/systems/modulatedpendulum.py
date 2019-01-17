import numpy as np
import matplotlib.pyplot as plt
from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *
from utils.quantum.husimi import *
import utils.plot.read as read
from utils.plot.latex import *

# This script contains:  4 classes and 4 functions

class PotentialMP(Potential):
	def __init__(self,e,gamma,f=np.cos):
		Potential.__init__(self)
		self.e=e
		self.gamma=gamma
		self.f=f # modulation waveform
		self.a1=getFourierCoefficient("a",1,self.f)
		self.b1=getFourierCoefficient("b",1,self.f)
		if gamma==0:
			self.d1=0.0
			self.x0=1.25
		else:
			self.d1=(self.gamma-0.25)/(self.e*self.gamma)
			self.x0=self.R1()
		self.isTimeDependent=True
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(1+self.e*self.f(t))*np.cos(x)
	
	def dVdx(self,x,t=np.pi/2.0):
		return self.gamma*(1+self.e*self.f(t))*np.sin(x)
		
	# The 4 following functions comes from classical analysis fo the bifurcation
	# they make possible to acess equilibrium positions for a given modulation waveform
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
		self.omega2= omega**2 #(grid.h*25.0)/(2*8113.9)
		
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
		
		
#~ class PotentialMPasym(PotentialHO):
	#~ # Adding longitudinal confinment to modulated pendulum
	#~ def __init__(self,e,gamma,x1,omega):
		#~ PotentialMP.__init__(self,e,gamma)
		#~ self.x1=x1
		#~ self.omega2= omega**2 #(grid.h*25.0)/(2*8113.9)
		
	#~ def Vx(self,x,t=np.pi/2.0):
		#~ return PotentialMP.Vx(self,x,t)+self.Vxasym(x)
	
	#~ def dVdx(self,x,t=np.pi/2.0):
		#~ return PotentialMP.Vx(self,x,t)+self.omega2*(x-self.x1)
		
	#~ def Vxasym(self,x):
		#~ # Returns the non-symetric contribution of potential
		#~ return 0.5*self.omega2*(x-self.x1)**2
		
	#~ def braketVxasym(self,wf1,wf2):
		#~ # Returns <wf1|Vxasym|wf2>
		#~ return sum(np.conj(wf1.x)*self.Vxasym(wf1.grid.x)*wf2.x)

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
	# WORK IN PROJECT
	def __init__(self,grid,potential,hermitian=False,idtmax=1,T0=1,g=0.0):
		self.potential=potential
		self.grid=grid
		self.dt=T0/idtmax
		
		self.g=g
		self.Up=np.zeros(grid.N,dtype=np.complex_)
		self.Up=np.exp(-(self.dt/self.grid.h)*(grid.p**2/2))
		
		self.muerrorref=1.0e-12
		
	def Ux(self, wfx):
		return np.exp(-(self.dt/self.grid.h)*(self.potential.Vx(self.grid.x)+self.g*abs(wfx)**2)/2.0)
		
	def getGroundState(self,wf):
		mudiff=1.0
		wf0x=wf.x
		mu=1.0

		i=0
		while mu > self.muerrorref:
			
			wf.x=wf.x*self.Ux(wf.x)
			wf.x2p()
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux(wf.x) 
			
			wf.normalizeX()
			
			mu=(1.0-abs(sum(np.conj(wf.x)*wf0x)*self.grid.intweight)**2)/self.dt

			if i%100==0:
				print("norm =",abs(wf%wf),"mudiff=",mu/self.muerrorref)
				
			i+=1
			wf0x=wf.x
		print("Converged in ",i,"iterations with dt=",self.dt)
		return wf	
				
class StrobosopicPhaseSpaceMP(StrobosopicPhaseSpace):
	def __init__(self,nperiod,ny0,timepropagator,pot,xmax=2*np.pi,pmax=2*np.pi):
		StrobosopicPhaseSpace.__init__(self,nperiod,ny0,timepropagator,xmax,pmax)
		self.pot=pot
		
		
	#~ def sety0(self,i):
		#~ dx=0.3
		#~ dp=0.1
		#~ self.y0=np.array([self.pot.R1()+rd.randint(0,101)/100.0*dx-0.5*dx,rd.randint(0,101)/100.0*dp-0.5*dp])
		
	#~ def sety0(self,i):
		#~ imax=self.ny0
		#~ print(i)
		#~ dx=2*np.pi
		#~ dp=3.0
		#~ if i<imax/4:
			#~ self.y0=np.array([-0.5*dx+i*2*dx/imax,0])
		#~ else:
			#~ i=i-imax/4
			#~ k=rd.randint(0,2)-1
			#~ print(k)
			#~ self.y0=np.array([k*2.5,-0.5*dp+i*2*dp/imax])
			
	def sety0(self,i,j):
		dx=2*np.pi
		dp=3.0
		dp=2*np.pi
		x0=dx*(-0.5+float(i+0.5)/self.ny0)
		p0=dp*(-0.5+float(j+0.5)/self.ny0)
		self.y0=np.array([x0,p0])
		print(x0,p0)
			
		
	def npz2plt(self, potential, datadir=""):
		R1=potential.R1()
		R2=potential.R2()
		thetaR1=potential.thetaR1()
		thetaR2=potential.thetaR2()
		#~ plt.scatter(R1*np.cos(thetaR1),0.5*R1*np.sin(thetaR1),s=3**2,marker="o",c="red")
		#~ plt.scatter(R2*np.cos(thetaR2),0.5*R2*np.sin(thetaR2),s=3**2,marker="o",c="red")
		#~ plt.scatter(R1*np.cos(thetaR1+np.pi),0.5*R1*np.sin(thetaR1+np.pi),s=3**2,marker="o",c="red")
		#~ plt.scatter(R2*np.cos(thetaR2+np.pi),0.5*R2*np.sin(thetaR2+np.pi),s=3**2,marker="o",c="red")
		StrobosopicPhaseSpace.npz2plt(self,datadir)

