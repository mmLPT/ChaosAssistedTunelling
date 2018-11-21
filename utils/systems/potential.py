import numpy as np
import matplotlib.pyplot as plt

class Potential:
	def __init__(self):
		pass
	def Vx(self,x):
		pass
	def mdVdx(self,x):
		pass
		
	@property
	def isTimeDependent(self):
		return self._isTimeDependent
		
	@isTimeDependent.setter
	def isTimeDependent(self, value):
		self._isTimeDependent = value
		
	@property
	def x0(self):
		return self._x0
		
	@x0.setter
	def x0(self, value):
		self._x0 = value
		
		
	def plot(self,xt,yt,dfile="potential",n=1000):
		x=np.linspace(xt[0],xt[1],n)
		plt.plot(x,self.Vx(x))
		ax = plt.axes()
		ax.set_xlim(xt[0],xt[1])
		ax.set_ylim(yt[0],yt[1])
		plt.savefig(dfile+".png", bbox_inches='tight')
		plt.clf() 
		

