import numpy as np
import matplotlib.pyplot as plt

class Potential:
	# The class potential is desgined to be herited only
	def __init__(self):
		self.isGP=False
		self.isTimeDependent=False
		self.g=0.0 # Gross-Pitaieskii strengh
		
		# Following may be note carefully
		# - for kicked system: T0=1.0, idtmax=1/dt
		# - for periodic system: T0 = period, idtmax=
		self.T0=0.0
		self.idtmax=0

	def Vx(self,x):
		pass
	def dVdx(self,x):
		pass
		
	def VGP(self,wfx):
		return self.g*abs(wfx)**2
		
	@property
	def isTimeDependent(self):
		return self._isTimeDependent
		
	@isTimeDependent.setter
	def isTimeDependent(self, value):
		self._isTimeDependent = value
		
	@property
	def isGP(self):
		return self._isGP
		
	@isGP.setter
	def isGP(self, value):
		self._isGP = value
		
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
		

