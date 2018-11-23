import numpy as np

from utils.quantum.grid import *
from utils.quantum.wavefunction import *

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
				wf.x2p()
				self.eigenvec.insert(i,wf)
		elif self.Mrepresentation=="p":
			for i in range(0,self.N):
				wf=WaveFunction(self.grid)
				wf.p=np.fft.ifftshift(eigenvec[:,i])*self.grid.phaseshift
				wf.p2x()
				self.eigenvec.insert(i,wf)
	
	def saveEvec(self,husimi,wdir,index=0):
		# This functions saves a given list of eigenvectors
		if index==0:
			index=range(0, self.N)
		for i in index:
			husimi.save(self.eigenvec[i],wdir+strint(i),title=str(np.angle(self.eigenval[i])))
			self.eigenvec[i].save(wdir+string(i))
				
class QuantumTimePropagator(QuantumOperator):
	# Class to be used to described time evolution operators such has
	# |psi(t')>=U(t',t)|psi(t)> with U(t',t)=U(dt,0)^idtmax
	# It relies on splliting method with H = p**2/2m + V(x,t)
	# It can be use for :
	# - periodic V(x,t) -> dt=T0/idtmax
	# - time-indepent V(x) -> 
	# - periodic kicked system : T0 = 1
	
	# /!\ Not adapted to non linear terms such a Gross-Pitaevskii
	def __init__(self,grid,potential,idtmax=1,T0=1):
		QuantumOperator.__init__(self,grid)
		self.hermitian=False
		
		self.potential=potential
		self.T0=T0 # Length of propagation
		self.idtmax=idtmax 
		self.dt=self.T0/self.idtmax
		
		# In order to gain time in propagation, we pre-compute 
		# splitted propagator that appears to be constant most of time
		self.Up=np.zeros(self.N,dtype=np.complex_)
		self.Up=np.exp(-(1j/grid.h)*(grid.p**2/4)*self.dt)
		self.Ux=np.zeros((idtmax,self.N),dtype=np.complex_)
		if self.potential.isTimeDependent:
			for idt in range(0,idtmax):
				self.Ux[idt]=np.exp(-(1j/grid.h)*(self.potential.Vx(grid.x,idt*self.dt))*self.dt)
		else:
			self.Ux[0]=np.exp(-(1j/grid.h)*(self.potential.Vx(grid.x))*self.dt)
			
	def propagate(self,wf):
		# Propagate over one period or over a 
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			wf.p=wf.p*self.Up  
			
	def fillM(self):
		# Propagate N dirac in x representation, to get matrix representation
		# of the quantum time propagator
		self.Mrepresentation="x"
		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			wf.setState("diracx",i0=i) 
			self.propagate(wf)
			wf.p2x()
			self.M[:,i]=wf.x 
			
class CATFloquetOperator(QuantumTimePropagator):
	# This class is specific for CAT purpose :
	def findTunellingStates(self,wf):
		# Find the two states that tunnels given a wavefunction
		
		# Check the overlap
		proj=np.zeros(self.N)
		for i in range(0,self.N):
			proj[i]=self.eigenvec[i]//wf
		
		# Find the two states that tunnel
		max1=np.argmax(proj)
		proj[max1]=0.0
		max2=np.argmax(proj)
		self.max1=max1
		self.max2=max2	
			
	def getSplitting(self):
		# Get the quasi-energy splitting and the tunneling period
		phi=abs(np.angle(self.eigenval[self.max1])-np.angle(self.eigenval[self.max2]))
		phi=min(phi,abs(phi-2*np.pi)) # Quasi energies are defined mod 2 pi
		T=2*np.pi/phi 
		qE=self.h*phi/self.T0
		return T, qE
		
	def saveTunellingStates(self,husimi,wdir):
		# Save husimi representation of tunneling states
		husimi.save(self.eigenvec[self.max1],wdir+"estate1",title=str(np.angle(self.eigenval[self.max1])))
		husimi.save(self.eigenvec[self.max2],wdir+"estate2",title=str(np.angle(self.eigenval[self.max2])))
