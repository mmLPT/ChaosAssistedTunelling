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
	
	# Following properties are x/p representation, they return set of
	# complex values corresponding to the givne representation
	# /!\ Calling x or p representation doesn't compute FFT
	@property   
	def x(self):
		return self._x
		
	@x.setter
	def x(self, value):
		self._x = value
        
	@property   
	def p(self):
		return self._p	

	@p.setter
	def p(self, value):
		self._p = value	
	
	def getx(self): # not tested
		self.p2x()
		return sum(self.grid.x*abs(self.x)**2)
		
	def getp2(self): # not tested
		return sum(self.grid.p**2*abs(self.p)**2)
	
	def normalize(self):
		nrm2=sum(abs(self.p)**2)
		self.p = self.p/np.sqrt(nrm2)
		self.p2x()
		
	def save(self,datafile):
		# Export both x/p representation in 'datafile.npz'
		self.p2x()
		np.savez(datafile,"w", x=self.grid.x, p=self.grid.p, psix=self.x, psip=np.fft.ifftshift(self.p))
	
	# Operators acting on objects
	def __add__(self,other): # wf1+wf2 
		# Add 2 wavefunctions
		psix=self.x+other.x
		wf=WaveFunction(self.grid)
		wf.setState("setx",psix=psix)
		return wf
		
	def __rmul__(self,scalar): # scalar*wf
		# Multiply by a scalar at left
		psix=self.x*scalar
		wf=WaveFunction(self.grid)
		wf.setState("setx",psix=psix)
		return wf
	
	def __mul__(self,scalar): # wf*scalar
		# Multiply by a scalar at right
		psix=self.x*scalar
		wf=WaveFunction(self.grid)
		wf.setState("setx",psix=psix)
		return wf
	
	def __truediv__(self,other): # wf1/wf2
		# Get braket <wf1|wf2>
		return sum(np.conj(self.x)*other.x)
		
	def __floordiv__(self,other): # wf1//wf2
		# Get probability |<wf1|wf2>|^2
		return abs(sum(np.conj(self.x)*other.x))**2
	
	# the two following functions call FTT algorithm and switch 
	# from x/p to p/x representation.
	def p2x(self):
		self.x=np.fft.ifft(self.p,norm="ortho")
		
	def x2p(self):
		self.p=np.fft.fft(self.x,norm="ortho") 
		
# = SET INITIAL STATE ================================================ #
# /!\ Numpy FFT algorithme works in [0,xmax] if you want to define a p
# wavefunction, make sure you center it, using self.grid.phaseshift

	def setState(self, state,x0=0.0,i0=0,psix=0,psip=0,xratio=1.0,datafile=""): # rajouter un truc du genre,wf=0): pour charge hevec pex
		if state=="coherent":
			# this gives a coherent state occupying a circled area in x/p
			# representation (aspect ratio), xration makes possible to 
			# contract the state in x direction
			sigma=xratio*np.sqrt(self.grid.h/2.0)
			self.x=np.exp(-(self.grid.x-x0)**2/(2*sigma**2))
			self.x2p()
			self.normalize()

		elif state=="diracx":
			# dirac in x at i0
			self.x=np.zeros(self.grid.N)
			self.x[i0]=1.0
			self.x2p()
			
		elif state=="diracp":
			# dirac in x at i0
			self.p=np.zeros(self.grid.N)
			self.p[i0]=1.0
			self.p=self.p*self.grid.phaseshift
			self.p2x()
			
		elif state=="setx":
			# loading
			self.x=np.zeros(self.grid.N,dtype=np.complex_)
			self.x=psix
			self.x2p()
			
		elif state=="setp":
			self.p=np.zeros(self.grid.N,dtype=np.complex_)
			self.p=psip
			self.p2x()
			
		elif state=="load":
			data=np.load(datafile+".npz")
			self.x=data['psix']
			self.p2x()
