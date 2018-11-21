import numpy as np
import matplotlib.pyplot as plt

# This script furnishes : one class

# ==================================================================== #
# Grid's class furnish an 1D grid in x and p, that can be used with 
# numpy FFT, wavefunction.
# One important thing to note is that numpy FFT work with both :
# ==================================================================== #

class Grid:
	def __init__(self,N,h,xmax=2*np.pi):
		self.N=N
		self.h=h
		self.xmax=xmax
		#~ self.x=(np.arange(self.N)-0.5*self.N)*self.xmax/self.N
		#~ self.dx=self.xmax/self.N
		#~ self.p=np.fft.fftfreq(self.N,self.dx)*2*np.pi*self.h 	
		#~ self.dp=0.5*h/self.dx
		#~ self.phaseshift=np.exp(-(1j/self.h)*(self.xmax/2.0)*self.p)
		self.x=np.linspace(-xmax/2.0,xmax/2.0,N)
		self.dx=self.xmax/self.N
		self.p=np.fft.fftfreq(self.N,self.dx)*2*np.pi*self.h 	
		self.dp=0.5*h/self.dx
		self.phaseshift=np.exp(-(1j/self.h)*(self.xmax/2.0)*self.p)
		
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
        
	@property   
	def N(self):
		return self._N
		
	@N.setter
	def N(self, value):
		self._N = value
        
	@property   
	def h(self):
		return self._h
		
	@h.setter
	def h(self, value):
		self._h = value
		
	@property   
	def xmax(self):
		return self._xmax
	
	@xmax.setter
	def xmax(self, value):
		self._xmax = value
		
	@property
	def phaseshift(self):
		return self._phaseshift
		
	@phaseshift.setter
	def phaseshift(self, value):
		self._phaseshift = value
