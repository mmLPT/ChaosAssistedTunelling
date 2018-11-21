import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *

class PotentialKR(Potential):
	def __init__(self,K):
		Potential.__init__(self)
		self.K=K
		
	def Vx(self,x):
		return self.K*np.cos(x)
	
	def dVdx(self,x):
		return -self.K*np.sin(x)
		
