import numpy as np
import matplotlib.pyplot as plt

class Potential:
	# The class potential is desgined to be herited only
	def __init__(self):	
		self.T0=1.0 
		self.idtmax=1
		self.x0=0.0

	def Vx(self,x):
		pass
	def dVdx(self,x):
		pass
		
	@property
	def T0(self):
		return self._T0
		
	@T0.setter
	def T0(self, value):
		self._T0 = value
		
	@property
	def idtmax(self):
		return self._idtmax
		
	@idtmax.setter
	def idtmax(self, value):
		self._idtmax = value
		
	@property
	def x0(self):
		return self._x0
		
	@x0.setter
	def x0(self, value):
		self._x0 = value
		

