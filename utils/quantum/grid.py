import numpy as np
import matplotlib.pyplot as plt

# This script contains : 1 class
# + class : grid

class Grid:
	# The Grid class provides a grid adapted to class WaveFunction
	# Id est, the p array is set to be FFT compatible and well
	# dimenzioned in h. All attributes are properties.
	def __init__(self,N,h,xmax=2*np.pi):
		self.h=h # hbar value
		
		self.N=N # Number of points in each space x/p
		self.xmax=xmax # Interval will be [-xmax2.0,xmax/2.0[
		self.x,self.dx=np.linspace(-xmax/2.0,xmax/2.0,N,endpoint=False,retstep=True)
		self.x=self.x+self.dx/2.0
		self.p=np.fft.fftfreq(self.N,self.dx)*2*np.pi*self.h 	
		self.dp=0.5*h/self.dx
		self.ddx=(np.max(self.x)-np.min(self.x))/(self.N-1)
		self.ddp=(np.max(self.p)-np.min(self.p))/(self.N-1)
		#print("é",np.max(self.x)-np.min(self.x),self.N,self.ddx)
		
		# A p-defined WaveFunction, don't know about x interval, only about 
		# it width, then to center a p-defined wf, you have to multiply
		# by the followinf factor
		self.phaseshift=np.exp(-(1j/self.h)*((self.xmax-self.dx)/2.0)*self.p)
	
	
	def getNforXshift(self,x0):
		# Return the index for shifting
		return int(x0/self.dx)
	
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
		
	@property
	def ddx(self):
		return self._ddx
		
	@ddx.setter
	def ddx(self, value):
		self._ddx = value

	@property
	def ddp(self):
		return self._ddp
		
	@ddp.setter
	def ddp(self, value):
		self._ddp = value
