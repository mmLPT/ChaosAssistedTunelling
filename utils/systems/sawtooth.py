import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *

class PotentialST(Potential):
	def __init__(self,alpha):
		Potential.__init__(self)
		self.alpha=alpha
		self.dx=1.0
		self.isTimeDependent=False
		self.T0=2*np.pi
		self.idtmax=1
		
	def Vx(self,x):
		return self.alpha*np.mod(x,self.dx)
	
	def dVdx(self,x):
		return -self.alpha
		
