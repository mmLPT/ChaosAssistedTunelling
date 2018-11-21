import numpy as np

from utils.quantum.grid import *

# ==================================================================== #
# Following class provides handy tools to manipulate wave function in  # 
# x/p space 
# ==================================================================== #

class WaveFunction:
	def __init__(self,grid):
		self.grid=grid
		self.x=np.zeros(grid.N,dtype=np.complex_)
		self.p=np.zeros(grid.N,dtype=np.complex_)
		
# = PROPERTIES ======================================================= #
# The two main properties that are acessible are x and p representation
# /!\ Calling x or p representation doesn't compute FFT, you might
# see "Changing representations"
		
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
		
	def __add__(self,other):
		psix=self.x+other.x
		wf=WaveFunction(self.grid)
		wf.setState("setx",psix=psix)
		return wf
	
	def __truediv__(self,other):
		return sum(np.conj(self.x)*other.x)
		
	def __floordiv__(self,other):
		return abs(sum(np.conj(self.x)*other.x))**2
		
# = CHANGING REPRESENTATION ========================================== #
# Whenever you need to update x/p representation you may use these     
# functions in order
	
	# the two following functions call FTT algorithm
	def p2x(self):
		self.x=np.fft.ifft(self.p,norm="ortho")
		
	def x2p(self):
		self.p=np.fft.fft(self.x,norm="ortho") 
		
# = OUTPUT =========================================================== #
# Export/get informations on wf 
	# not tested	
	def getx(self):
		self.p2x()
		return sum(self.grid.x*abs(self.x)**2)
		
	# not tested	
	def getp2(self):
		return sum(self.grid.p**2*abs(self.p)**2)
		
# = OUTPUT =========================================================== #	
	# this save the actual wf in a given directory
	def save(self,path):
		self.p2x()
		np.savez(path,"w", x=self.grid.x, p=self.grid.p, psix=self.x, psip=np.fft.ifftshift(self.p))
		
# = SET INITIAL STATE ================================================ #
# /!\ Numpy FFT algorithme works in [0,xmax] if you want to define a p
# wavefunction, make sure you center it, using self.grid.phaseshift

	def setState(self, state,x0=0.0,i0=0,psix=0,psip=0,xratio=1.0): # rajouter un truc du genre,wf=0): pour charge hevec pex
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

	def normalize(self):
		nrm2=sum(abs(self.p)**2)
		self.p = self.p/np.sqrt(nrm2)
		self.p2x()
		
	def setAmplitude(self,amplitude):
		self.x=self.x*amplitude
		self.x2p()
