import numpy as np
import matplotlib.pyplot as plt

from utils.quantum.grid import *
from utils.quantum.quantumoperator import *
from utils.classical.stroboscopic import *
from utils.mathtools.periodicfunctions import *
from utils.systems.potential import *



class PotentialDW(Potential):
	def __init__(self,e,gamma,f=np.cos,idtmax=1000):
		Potential.__init__(self)
		self.T0=4*np.pi
		self.idtmax=idtmax
		self.x0=np.pi/2
		self.e=e
		self.gamma=gamma
		self.f=f 
		
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(1+self.e*self.f(t))*(x**2-self.x0**2)*x**2/(np.pi)**3
	
	def dVdx(self,x,t=np.pi/2.0):
		return self.gamma*(1+self.e*self.f(t))*(2*x**2-self.x0**2)*2*x/(np.pi)**3
	
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

	
