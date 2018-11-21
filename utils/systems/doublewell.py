import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *



class PotentialDW(Potential):
	def __init__(self,x0,alpha0,x1,alpha1):
		Potential.__init__(self)
		self.alpha0=alpha0
		self.x0=x0
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
		pot0=self.alpha0*(((x%self.Tx)-self.Tx/2.0)**2-self.x0**2)**2
		pot1=self.alpha1*(x-self.x1)
		return pot0+pot1
	
	def dVdx(self,x):
		dpot0=4*((x%self.Tx)-self.Tx/2.0)*self.alpha0*(((x%self.Tx)-self.Tx/2.0)**2-self.x0**2)
		dpot1=self.alpha1
		return dpot0+dpot1

	
