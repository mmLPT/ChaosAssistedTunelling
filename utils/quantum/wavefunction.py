import numpy as np

from utils.quantum.grid import *

# This script contains : 1 class
# + class : WaveFunction

class WaveFunction:
	# The wavefunction class is used to describe a state in a 1D 
	# infinite Hiblert space. It provides x and p representation.
	# One needs to switch by hand representation every time it's needed
	
	def __init__(self,grid):
		self.grid=grid
		self.x=np.zeros(grid.N,dtype=np.complex_)
		self.p=np.zeros(grid.N,dtype=np.complex_)
	
	# === Getters/Setters ==============================================
	@property   
	def x(self):
		# Get <x|psi> 
		return self._x
		
	@x.setter
	def x(self, value):
		# Set <x|psi> from complex N-array
		self._x = value
     
	@property   
	def p(self):
		# Get <p|psi> 
		return self._p	

	@p.setter
	def p(self, value):
		# Set <p|psi> from complex N-array
		self._p = value	
		
	@property   
	def grid(self):
		return self._grid	

	@grid.setter
	def grid(self, value):
		self._grid = value	
	
	def setState(self, state,x0=0.0,i0=0,psix=0,psip=0,xratio=1.0,datafile=""): 
		# Commons physical states are implemented
		if state=="coherent":
			# This gives a coherent state occupying a circled area in x/p 
			# representation (aspect ratio), xratio makes possible to 
			# contract the state in x direction
			sigma=xratio*np.sqrt(self.grid.h/2.0)
			self.x=np.exp(-(self.grid.x-x0)**2/(2*sigma**2))
			self.normalizeX()
			self.x2p()

		elif state=="diracx":
			# Set <x|psi> = delta(x-x[i0])
			self.x=np.zeros(self.grid.N)
			self.x[i0]=1.0
			self.x2p()
			
		elif state=="diracp":
			# Set <p|psi> = delta(p-p[i0])
			self.p=np.zeros(self.grid.N)
			self.p[i0]=1.0
			self.p=self.p*self.grid.phaseshift
			self.p2x()
			
		elif state=="load":
			# Set |psi> from a file
			data=np.load(datafile+".npz")
			self.x=data['psix']
			self.x2p()
			
	def normalizeX(self):
		# Normalize <x|psi>
		nrm=abs(sum(np.conj(self.x)*self.x))
		self.x = self.x/np.sqrt(nrm)
		
	def normalizeP(self):
		# Normalize <p|psi>
		nrm=abs(sum(np.conj(self.p)*self.p))
		self.p = self.p/np.sqrt(nrm)
		
	# === Switching representation x <-> p =============================
	def p2x(self):
		# <p|psi> -> <x|psi>
		self.x=np.fft.ifft(self.p,norm="ortho")
		
	def x2p(self):
		# <x|psi> -> <p|psi>
		self.p=np.fft.fft(self.x,norm="ortho") 
	
	# === Operations on wave function ==================================
	def __add__(self,other): 
		# wf1+wf2 <-> |wf1>+|wf2>
		psix=self.x+other.x
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
		
	def __sub__(self,other): 
		# wf1+wf2 <-> |wf1>+|wf2>
		psix=self.x-other.x
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
		
	def __rmul__(self,scalar): 
		# a*wf <-> a|wf>
		psix=self.x*scalar
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
	
	def __mul__(self,scalar):
		# wf*a <-> a|wf>
		psix=self.x*scalar
		wf.x=psix
		wf.x2p()
		return wf
		
	def __truediv__(self,scalar): 
		# wf/a <-> |wf>/a
		psix=self.x/scalar
		wf.x=psix
		wf.x2p()
		return wf
	
	def __mod__(self,other): 
		# wf1%wf2 <-> <wf1|wf2>
		return sum(np.conj(self.x)*other.x)
		
	def __floordiv__(self,other): 
		# wf1//wf2 <-> |<wf1|wf2>|^2
		return abs(sum(np.conj(self.x)*other.x))**2
		
	# === I/O ==========================================================
	def isSymetricInX(self,sigma=0.01):
		mwf=WaveFunction(self.grid)
		psix=np.zeros(self.grid.N,dtype=np.complex_)
		for i in range(0,self.grid.N):
			psix[i]=self.x[self.grid.N-1-i]
		mwf.x=psix
		if sum(abs((mwf-self).x)**2) < sigma:
			return True
		else:
			return False
	
	def getx(self): 
		# Get <psi|x|psi>
		return sum(self.grid.x*abs(self.x)**2)
		
	def getp2(self): 
		# Get <psi|p^2|psi>
		return sum(self.grid.p**2*abs(self.p)**2)
	
	def save(self,datafile):
		# Export both x/p representation in 'datafile.npz'
		np.savez(datafile,"w", x=self.grid.x, p=self.grid.p, psix=self.x, psip=np.fft.ifftshift(self.p))
