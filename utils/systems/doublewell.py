import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *



class PotentialDW(Potential):
	def __init__(self,alpha0,x1,alpha1):
		Potential.__init__(self)
		self.alpha0=alpha0
		self.x0=np.pi/2.0
		self.alpha1=alpha1
		self.x1=x1
		self.Tx=2*np.pi #spatial period
		self.isTimeDependent=False
		
	@property
	def x0(self):
		return self._x0
		
	@x0.setter
	def x0(self, value):
		self._x0 = value
		
	def Vx(self,x):
		return self.alpha0*(self.xmodt(x)**2-self.x0**2)**2+self.Vxasym(x)
		
	def dVdx(self,x):
		return 2*self.alpha0*(self.xmod(x)**2-self.x0**2)*2*self.xmod(x)+self.dVdxasym(x)
	
	def Vxasym(self,x):
		return self.alpha1*(x-self.x1)
		
	def dVdxasym(self,x):
		return self.alpha1
		
	def xmod(self,x):
		x=x%self.Tx
		if x>self.Tx/2.0:
			x=x-self.Tx
		return x
	
	def xmodt(self,x):
		for i in range(0,x.shape[0]):
			x[i]=self.xmod(x[i])
		return x

	
