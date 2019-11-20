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
		
	# ~ def Vx(self,x):
		# ~ return -4*np.pi**2*self.gamma*(np.cos(x)*np.array(x>0)-np.cos(x)*np.array(x<0))
		
class PotentialKR(Potential):
	def __init__(self,K):
		Potential.__init__(self)
		self.K=K
		self.T0=1.0
		self.idtmax=1
		
	def Vx(self,x):
		return self.K*np.cos(x)
		
class PotentialGG(Potential):
	def __init__(self,a,V0):
		Potential.__init__(self)
		self.a=a
		self.V0=V0
		self.T0=1.0
		self.idtmax=1
		
	def Vx(self,x):
		return -2*np.pi*self.V0*np.array(x>-self.a)*np.array(x<self.a)
		
