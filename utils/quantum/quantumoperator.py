import numpy as np

from utils.quantum.grid import *
from utils.quantum.wavefunction import *

# This script contains: 3 classes
# + class : QuantumOperator
# + class : QuantumTimePropagator
# + class : CATFloquetOperator

class QuantumOperator:
# This class provides a discrete representation for an x/p acting op.  
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
		# has to be constructed in any derivated class
		pass
					
	def diagonalize(self):
		# routine to diagonalyze the representation
		# and construct eigenstates as
		self.fillM()
		eigenvec=np.zeros((self.N,self.N),dtype=np.complex_)
		if self.hermitian:
			self.eigenval,eigenvec=np.linalg.eigh(self.M)
		else:
			self.eigenval,eigenvec=np.linalg.eig(self.M)
			
		if self.Mrepresentation=="x": 
			for i in range(0,self.N):
				wf=WaveFunction(self.grid)
				wf.setState("setx",psix=eigenvec[:,i])
				self.eigenvec.insert(i,wf)
		elif self.Mrepresentation=="p":
			for i in range(0,self.N):
				wf=WaveFunction(self.grid)
				wf.setState("setp",psip=np.fft.ifftshift(eigenvec[:,i])*self.grid.phaseshift)
				self.eigenvec.insert(i,wf)

	def saveEvec(self,husimi,wdir,index=0):
		# This functions saves a given list of eigenvectors
		if index==0:
			index=range(0, self.N)
		for i in index:
			husimi.save(self.eigenvec[i],wdir+"{:05d}".format(i),title=str(np.angle(self.eigenval[i])))
			#self.eigenvec[i].save(wdir+"/wf/{:05d}".format(i))
				
class QuantumTimePropagator(QuantumOperator):
	# Time evolution operators
	# Class adapted to time indepedent Hamilotnian/periodic one
	def __init__(self,grid,potential,hermitian=False,idtmax=1,T0=1):
		QuantumOperator.__init__(self,grid)
		
		self.potential=potential
		
		self.hermitian=hermitian
		self.T0=T0
		self.idtmax=idtmax
		self.dt=self.T0/self.idtmax
		
		
		self.Up=np.zeros(self.N,dtype=np.complex_)
		self.Up=np.exp(-(1j/grid.h)*(grid.p**2/4)*self.dt)
		self.Ux=np.zeros((idtmax,self.N),dtype=np.complex_)
		if self.potential.isTimeDependent:
			for idt in range(0,idtmax):
				self.Ux[idt]=np.exp(-(1j/grid.h)*(self.potential.Vx(grid.x,idt*self.dt))*self.dt)
		else:
			self.Ux[0]=np.exp(-(1j/grid.h)*(self.potential.Vx(grid.x))*self.dt)
			
	def propagate(self,wf):
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			wf.p=wf.p*self.Up  
			
	def fillM(self):
		self.Mrepresentation="x"
		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			wf.setState("diracx",i0=i) 
			self.propagate(wf)
			wf.p2x()
			self.M[:,i]=wf.x 
			
class CATFloquetOperator(QuantumTimePropagator):
	def findTunellingStates(self,wf):
		proj=np.zeros(self.N)
		
		for i in range(0,self.N):
			proj[i]=self.eigenvec[i]//wf
			
		max1=np.argmax(proj)
		proj[max1]=0.0
		max2=np.argmax(proj)
		self.max1=max1
		self.max2=max2	
			
	def getSplitting(self):
		phi=abs(np.angle(self.eigenval[self.max1])-np.angle(self.eigenval[self.max2]))
		phi=min(phi,abs(phi-2*np.pi))
		T=2*np.pi/phi
		qE=self.h*phi/self.T0
		return T, qE
		
	def saveTunellingStates(self,husimi,wdir):
		husimi.save(self.eigenvec[self.max1],wdir+"estate1",title=str(np.angle(self.eigenval[self.max1])))
		husimi.save(self.eigenvec[self.max2],wdir+"estate2",title=str(np.angle(self.eigenval[self.max2])))
