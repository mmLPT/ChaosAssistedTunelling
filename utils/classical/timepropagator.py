import numpy as np

from utils.mathtools.RK4 import *

# This script contains : 2 classes
# + class : ClassicalDiscreteTimePropagator
# + class : ClassicalContinueTimePropagator

class ClassicalDiscreteTimePropagator:
	def __init__(self,potential):
		self.potential=potential

	def propagate(self,y):
		yp=np.zeros(2) #y'
		# ~ yp[1]=y[1]-self.potential.dVdx(y[0])
		# ~ yp[0]=y[0]+yp[1]
		yp[1]=y[1]-self.potential.dVdx(y[0]+0.5*y[1])
		yp[0]=y[0]+0.5*(yp[1]+y[1])
		return yp
	
class ClassicalContinueTimePropagator(ClassicalDiscreteTimePropagator):
	def __init__(self,potential,tstep=100,dt=2*np.pi/100):
		ClassicalDiscreteTimePropagator.__init__(self,potential)
		self.tstep=tstep
		self.dt=dt
		

	def f(self,y,t):
		# scheme to solve motion equation with RK4 such as y'=f(y,t) with y at 2D
		yp=np.zeros(2) 
		yp[0]=y[1]
		yp[1]=-self.potential.dVdx(y[0],t)
		return yp

	def propagate(self,y):		
		for i in range(0,self.tstep):
			y=RK4(self.f,y,i*self.dt,self.dt) #propagation
		return y

