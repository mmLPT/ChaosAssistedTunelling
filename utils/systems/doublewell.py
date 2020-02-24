import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *



class PotentialDW(Potential):
	def __init__(self,gamma,idtmax=1000):
		Potential.__init__(self)
		self.T0=1
		self.idtmax=idtmax
		self.x0=np.pi/2
		self.gamma=gamma
		
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(np.cos(x)-np.cos(2*x))
	
	def dVdx(self,x,t=np.pi/2.0):
		return self.gamma*(np.sin(x)-2*np.sin(2*x))
	
	# ~ def Vxasym(self,x):
		# ~ return self.alpha1*(x-self.x1)
		
	# ~ def dVdxasym(self,x):
		# ~ return self.alpha1
		
	# ~ def xmod(self,x):
		# ~ x=x%self.Tx
		# ~ if x>self.Tx/2.0:
			# ~ x=x-self.Tx
		# ~ return x
	
	# ~ def xmodt(self,x):
		# ~ for i in range(0,x.shape[0]):
			# ~ x[i]=self.xmod(x[i])
		# ~ return x

	
