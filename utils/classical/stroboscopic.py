import matplotlib.pyplot as plt
import numpy as np
import random as rd
from utils.classical.timepropagator import *
from utils.toolsbox import *
from utils.plot.latex import *

class StrobosopicPhaseSpace:
	# This class can be used to generate a stroposcopic phase space of a 
	# given time periodic system. It requires time propagator, to include 
	# both discrete and continous time map.
	def __init__(self,nperiod,ny0,timepropagator,xmax=2*np.pi,pmax=2*np.pi):
		self.nperiod=nperiod
		self.ny0=ny0
		self.y0=np.zeros(2)
		self.xmax=xmax
		self.pmax=pmax
		self.timepropagator=timepropagator
	
	def getTrajectory(self,i,j):
		# this fonciton computes strobsoocpic trajectory over nperiod 
		# for a given initial state y0
		self.sety0(i,j)
		y=self.y0
		xs=np.zeros(self.nperiod)
		ps=np.zeros(self.nperiod)
		for k in range(0,self.nperiod):
			y=self.timepropagator.propagate(y)
				
			xs[k]=y[0]%(self.xmax)
			if xs[k]>self.xmax/2.0:
				xs[k]=xs[k]-self.xmax
			ps[k]=y[1]
		
		return xs, ps
	
	def save(self,wdir=""):
		for i in range(0,self.ny0):
			for j in range(0,self.ny0):
				xs,ps = self.getTrajectory(i,j)
				np.savez(wdir+strint(i*self.ny0+j),"w", x=xs, p=ps)
			
	def npz2plt(self, wdir=""):
		setLatex()
		# Read .npz file and print it
		ax = plt.axes()
		ax.set_xlim(-self.xmax/2.0,self.xmax/2.0)
		ax.set_ylim(-self.pmax/2.0,self.pmax/2.0)
		#ax.set_ylim(-1.0,1.0)
		ax.set_aspect('equal')
		for i in range(0,self.ny0):
			for j in range(0,self.ny0):
				data=np.load(wdir+strint(i*self.ny0+j)+".npz")
				x=data["x"]
				p=data["p"]
				plt.scatter(x,p,s=0.1**2,c="black")
		plt.show()
		#saveLatex("classicalPS")
		
	def sety0(self,i,j):
		self.y0=np.array([rd.randint(0,101)/100.0*self.xmax-self.xmax/2.0,rd.randint(0,101)/100.0*self.pmax-self.pmax/2.0])
