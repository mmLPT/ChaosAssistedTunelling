import numpy as np

from utils.quantum.grid import *
from utils.quantum.wavefunction import *
from numpy.linalg import matrix_power

# This script contains: 3 classes
# + class : QuantumOperator
# + class : QuantumTimePropagator
# + class : CATFloquetOperator

class QuantumOperator:
	# This class provides a discrete representation for an x/p acting operator
	def __init__(self, grid):
		self.grid=grid
		self.N=grid.N
		self.h=grid.h
		self.hermitian=False
		self.Mrepresentation="x" #x or p depending on how you fillup M
		self.M=np.zeros((self.N,self.N),dtype=np.complex_) # The operator representation
		self.eigenval=np.zeros(self.N,dtype=np.complex_) 
		self.eigenvec=[] # Eigenstates (from wave function class) id est can be used in both representation
	
	def fillM(self):
		# Fill the matrix representation of the operator
		pass
					
	def diagonalize(self):
		# Diagonalyze the matrix representation of the operator and 
		# save the wavefunctions
		self.fillM()
		eigenvec=np.zeros((self.N,self.N),dtype=np.complex_)
		if self.hermitian:
			self.eigenval,eigenvec=np.linalg.eigh(self.M)
		else:
			self.eigenval,eigenvec=np.linalg.eig(self.M)
			
		if self.Mrepresentation=="x": 
			for i in range(0,self.N):
				wf=WaveFunction(self.grid)
				wf.x=eigenvec[:,i]
				wf.normalizeX()
				wf.x2p()
				self.eigenvec.insert(i,wf)
		elif self.Mrepresentation=="p":
			for i in range(0,self.N):
				wf=WaveFunction(self.grid)
				wf.p=np.fft.ifftshift(eigenvec[:,i])*self.grid.phaseshift
				wf.p2x()
				wf.normalizeX()
				wf.x2p()
				self.eigenvec.insert(i,wf)
	
	def saveEvec(self,husimi,wdir,index=0):
		# This functions saves a given list of eigenvectors
		if index==0:
			index=range(0, self.N)
		for i in index:
			husimi.save(self.eigenvec[i],wdir+strint(i),title=str(np.angle(self.eigenval[i])))
			self.eigenvec[i].save(wdir+string(i))

	def getEvec(self,i):
		return self.eigenvec[i]
		
	def getEval(self,i):
		return self.eigenval[i]
		
	@property
	def M(self):
		return self._M
		
	@M.setter
	def M(self, value):
		self._M = value
				
class QuantumTimePropagator(QuantumOperator):
	# Class to be used to described time evolution operators such has
	# |psi(t')>=U(t',t)|psi(t)> with U(t',t)=U(dt,0)^idtmax
	# It relies on splliting method with H = p**2/2m + V(x,wfx,t)
	# It can be use for :
	# - periodic V(x,t) 
	# - time-indepent V(x)
	# - periodic kicked system V(x,t)
	# - non linear GP V(x,wfx,t)

	def __init__(self,grid,potential,beta=0.0,mu=1.0,randomphase=False):
		QuantumOperator.__init__(self,grid)
		self.hermitian=False
		
		self.potential=potential
		self.T0=potential.T0 # Length of propagation
		self.idtmax=potential.idtmax # number of step : 1 -> kicked/ =/=1 -> periodic or time independent
		self.dt=self.T0/self.idtmax # time step
		self.beta=beta # quasi-momentum
		self.mu=mu # 'mass'
		
		# In order to gain time in propagation, we pre-compute 
		# splitted propagator that appears to be constant most of time
		# NB: if you have non interactions terms, this doesn't work
		self.Up=np.zeros(self.N,dtype=np.complex_)
		if randomphase:
			self.Up=np.exp(-1j*(np.random.rand(self.N)*2*np.pi)/2.0*self.dt)
		else:
			self.Up=np.exp(-(1j/grid.h)*((grid.p-self.beta)**2/(self.mu*4))*self.dt)
		

		self.Ux=np.zeros((self.idtmax,self.N),dtype=np.complex_)
		if self.idtmax==1:
			for idt in range(0,self.idtmax):
				self.Ux[idt]=np.exp(-(1j/grid.h)*(self.potential.Vx(grid.x))*self.dt)
		else:
			for idt in range(0,self.idtmax): 
				self.Ux[idt]=np.exp(-(1j/grid.h)*(self.potential.Vx(grid.x,idt*self.dt))*self.dt)
			
	def propagate(self,wf):
		# Propagate over one period/kick/arbitray time 
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			wf.p=wf.p*self.Up  
			
	def propagateGP(self,wf,g):
		# Propagate over one period/kick/arbitray time with interactions
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]*np.exp(-(1j/self.grid.h)*(g*np.abs(wf.x)**2)*self.dt)
			wf.x2p() 
			wf.p=wf.p*self.Up  
			
	def fillM(self):
		# Propagate N dirac in x representation, to get matrix representation
		# of the quantum time propagator
		self.Mrepresentation="x"
		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			wf.setState("diracx",i0=i,norm=False) #Norm false to produce 'normalized' eigevalue
			self.propagate(wf)
			wf.p2x()
			self.M[:,i]=wf.x 
			
class CATFloquetOperator(QuantumTimePropagator):
	# This class is specific for CAT purpose:
	# WIP: a bit dirty.
	def __init__(self,grid,potential,beta=0.0,randomphase=False):
		QuantumTimePropagator.__init__(self,grid,potential,beta=beta,randomphase=randomphase)
		self.qE=np.zeros(grid.N) # quasi energies
		
	def diagonalize(self):
		# Diagonalize, then compute quasi-energies
		QuantumOperator.diagonalize(self)
		for i in range(0,self.N):
			self.qE[i]=-np.angle(self.eigenval[i])*(self.h/self.T0)	
	
	def getOrderedOverlapsWith(self,wf,twolvlonly=False):
		# Check overlaps with a given wave function
		# Returns the index of ordered overlaps and the overlaps
		overlaps=np.zeros(self.N,dtype=complex)
		for i in range(0,self.N):
			overlaps[i]=self.eigenvec[i]%wf
		if twolvlonly==True:
			i1=np.argmax(self.overlaps)
			proj1=overlaps[i1]
			overlaps[i1]=0.0
			i2=np.argmax(self.overlaps)
			return i1,i2,proj1,proj2
		else:
			ind=np.flipud(np.argsort(np.abs(overlaps)**2))
			orderedOverlaps=np.zeros(self.N,dtype=complex)
			for i in range(0,self.N):
				orderedOverlaps[i]=overlaps[ind[i]]
			return ind, orderedOverlaps
			
	def getTunnelingPeriodBetween(self,i1,i2):		
		return 2*np.pi*self.h/(self.T0*(abs(self.diffqE1qE2(i1,i2))))
		
	def getTunnelingFrequencyBetween(self,i1,i2):		
		return np.abs(self.diffqE1qE2(i1,i2))/self.h
		
	def getQE(self,i0):
		# Returns either the quasi-energy of quasi-ground state or quasi-first excited state
		return self.qE[i0]
			
	def getQETh(self,i0,pot):
		# Returns expected value of quasi-energies according to 
		# perturbation theory up to 3rd order for a given potential
		V00=pot.braketVxasym(self.eigenvec[0],self.eigenvec[0])
		V11=pot.braketVxasym(self.eigenvec[1],self.eigenvec[1])
		V01=pot.braketVxasym(self.eigenvec[0],self.eigenvec[1])
		V10=pot.braketVxasym(self.eigenvec[1],self.eigenvec[0])
		E0mE1=self.diffqE1qE2(0,1)
		E1mE0=-E0mE1
		
		if i0==0:
			e0=self.qE[self.iqgs]
			e1=abs(V00)
			e2=abs(V01)**2/E0mE1
			e3=abs(V01)**2/(E0mE1)**2*(abs(V11)-abs(V00))
			#e4=abs(V01)**2*abs(V11)**2/E0mE1**3-e2*abs(V10)**2/E0mE1**4-2*abs(V00)*abs(V01)**2*abs(V11)/E0mE1**3+abs(V00)**2*abs(V01)**2/E0mE1**3
		elif i0==1:
			e0=self.qE[self.iqfes]
			e1=abs(V11)
			e2=abs(V01)**2/E1mE0
			e3=abs(V01)**2/(E0mE1)**2*(abs(V00)-abs(V11))
			#e4=abs(V11)**2*abs(V00)**2/E1mE0**3-e2*abs(V10)**2/E1mE0**4-2*abs(V11)*abs(V01)**2*abs(V00)/E1mE0**3+abs(V11)**2*abs(V01)**2/E1mE0**3
		
		e=e0 +e1 +e2 #+e3
		return e	
		
	def diffqE1qE2(self,i1,i2):
		# This returns the difference on a circle
		qE1=self.qE[i1]
		qE2=self.qE[i2]
		dE=np.pi*(self.h/self.T0)
		diff=qE1-qE2
		if diff>dE:
			return diff-2*dE
		elif diff<-dE:
			return diff+2*dE
		else:
			return diff
			
	def getSpacingDistribution(self):
		ind=np.argsort(self.qE)
		# On suppose Nh=2pi et T=2pi do we
		s=np.zeros(self.N)
		for i in range(0,self.N-1):
			s[i]=np.abs(self.diffqE1qE2(ind[i],ind[i+1]))/(2*np.pi*self.h/(self.N*self.T0))
		s[self.N-1]=np.abs(self.diffqE1qE2(ind[self.N-1],ind[0]))/(2*np.pi*self.h/(self.N*self.T0))
		return s
		
	def getFormFactor(self,it):
		n=int(it*self.N)
		return abs(sum(self.eigenval**n))**2/self.N			
		
# Work In Progress # ------------------------------------------------- #

# ~ class QuantumImaginaryTimePropagator(QuantumOperator):
	# ~ # This class is used to find the ground state of a given potential
	# ~ # Note that is mean to be used for GP + no time dependent potential
	
	# ~ def __init__(self,grid,potential,idtmax):
		# ~ self.potential=potential
		# ~ self.grid=grid
		
		# ~ self.T0=potential.T0 
		# ~ self.idtmax=idtmax 
		# ~ self.dt=self.T0/self.idtmax
		
		# ~ self.Up=np.zeros(grid.N,dtype=np.complex_)
		# ~ self.Up=np.exp(-(self.dt/self.grid.h)*(grid.p**2/2))
		
		# ~ self.muerrorref=1.0e-14
		
	# ~ def Ux(self, wfx):
		# ~ # Split step x propagator
		# ~ return np.exp(-(self.dt/self.grid.h)*(self.potential.Vx(self.grid.x,wfx))/2.0)
		
	# ~ def getGroundState(self,wf):
		# ~ # Initialization
		# ~ wf0x=wf.x
		# ~ mu=1.0
		# ~ i=0
	
		# ~ # Propagation
		# ~ while mu > self.muerrorref:
			
			# ~ # Split step method
			# ~ wf.x=wf.x*self.Ux(wf.x)
			# ~ wf.x2p()
			# ~ wf.p=wf.p*self.Up 
			# ~ wf.p2x() 
			# ~ wf.x=wf.x*self.Ux(wf.x) 
			
			# ~ # Notmalization
			# ~ wf.normalizeX()
			
			# ~ # Compute mu=(1-||<wf(t)|wf(t+dt)>||^2)/dt to compare with monitoring value
			# ~ mu=(1.0-abs(sum(np.conj(wf.x)*wf0x)*self.grid.intweight)**2)/self.dt
			
			# ~ # Iteration
			# ~ wf0x=wf.x
			# ~ i+=1
			
			# ~ # Output
			# ~ if i%100==0:
				# ~ print("norm =",abs(wf%wf),"mudiff=",mu/self.muerrorref)
				
		# ~ # Converged
		# ~ print("Converged in ",i,"iterations with dt=",self.dt)
		# ~ return wf

