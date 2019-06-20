import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *

class PotentialST(Potential):
	def __init__(self,gamma):
		Potential.__init__(self)
		self.gamma=gamma
		self.dx=2*np.pi
		self.T0=1.0
		self.idtmax=1
		
	def Vx(self,x):
		return -4*np.pi**2*self.gamma*np.mod(x/self.dx,1.0)
		
