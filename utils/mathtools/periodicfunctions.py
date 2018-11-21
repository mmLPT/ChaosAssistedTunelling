import numpy as np
import random as rd
from utils.mathtools.simpsonintegrator import *

# This provides a set of various periodic function, by default they are
# centered in 0 and varies between +/-1
# You also can find a function to compute any of fourier coefficient

class PeriodicFunctions:
	def __init__(self,phase=0.0,amplitude=1.0,T=2*np.pi,offset=0.0):
		self.phase=phase
		self.amplitude=amplitude
		self.T=T
		self.offset=offset
		
	def gettime(self,t):
		return (t%self.T)+self.phase
		
	def triangle(self,t): #triangle a1=-8/pi^2
		v=0.0
		t=self.gettime(t)
		alpha=4.0/self.T
		if t<self.T/2.0:
			v=alpha*t-1.0
		else:
			v=-alpha*t+3.0
		return v*self.amplitude
		
	def square(self,t): # carre a1=0 b1=4/pi
		v=0.0
		t=self.gettime(t)
		if t<(self.T/2.0):
			v=1.0
		else:
			v=-1.0
		return v*self.amplitude
		
	def sawtooth(self,t): #dent de scie a1=0 b1=2/pi
		v=0.0
		t=self.gettime(t)
		alpha=2.0/self.T
		v=alpha*t-1.0
		return v*self.amplitude
		
	def cos(self,t):
		t=self.gettime(t)
		return np.cos(t)*self.amplitude
		
def getFourierCoefficient(ab,n,f,T=2*np.pi,npoints=100000):
	fc=0.0
	if ab == "a":
		def a2int(t):
			return 2.0/T*f(t)*np.cos(n*t*2*np.pi/T)
		fc=simpsonIntegrate(npoints,-T/2.0,T/2.0,a2int)
	elif ab == "b":
		def b2int(t):
			return 2.0/T*f(t)*np.sin(n*t*2*np.pi/T)
		fc=simpsonIntegrate(npoints,-T/2.0,T/2.0,b2int)
	return fc
	
