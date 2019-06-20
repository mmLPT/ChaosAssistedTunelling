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
	def __init__(self,nperiod,ny02,timepropagator,xmax=2*np.pi,pmax=2.0,random=False):
		self.nperiod=nperiod
		self.ny0=int(np.sqrt(ny02))
		self.random=random
		self.xmax=xmax
		self.pmax=pmax
		self.dx=self.xmax/self.ny0
		self.dp=self.pmax/self.ny0
		self.timepropagator=timepropagator
	
	def getTrajectory(self,i):
		# this fonciton computes strobsoocpic trajectory over nperiod 
		# for a given initial state y0
		y=self.generatey0(i)
		print(i,y)
		xs=np.zeros(self.nperiod)
		ps=np.zeros(self.nperiod)
		for k in range(0,self.nperiod):
			y=self.timepropagator.propagate(y)
				
			xs[k]=y[0]%(self.xmax)
			if xs[k]>self.xmax/2.0:
				xs[k]=xs[k]-self.xmax
			ps[k]=y[1]
		
		return xs, ps
		
	def getOrbit(self,y0):
		# this fonciton computes strobsoocpic trajectory over nperiod 
		# for a given initial state y0
		y=y0
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
				xs,ps = self.getTrajectory(i*self.ny0+j)
				np.savez(wdir+strint(i*self.ny0+j),"w", x=xs, p=ps)
			
	def npz2plt(self, wdir):
		# Read .npz file and print it
		ax = plt.axes()
		ax.set_xlim(-self.xmax/2.0,self.xmax/2.0)
		ax.set_ylim(-self.pmax/2.0,self.pmax/2.0)
		ax.set_aspect('equal')
		for i in range(0,self.ny0**2-1):
			data=np.load(wdir+strint(i)+".npz")
			x=data["x"]
			p=data["p"]
			plt.scatter(x,p,s=01.0**2)
		plt.show()

	def npz2png(self, wdir):
		# Read .npz file and print it
		ax = plt.axes()
		ax.set_xlim(-self.xmax/2.0,self.xmax/2.0)
		ax.set_ylim(-self.pmax/2.0,self.pmax/2.0)
		ax.set_aspect('equal')
		for i in range(0,self.ny0**2-1):
			data=np.load(wdir+strint(i)+".npz")
			x=data["x"]
			p=data["p"]
			plt.scatter(x,p,s=01.0**2)
		plt.savefig(wdir+"SPS.png")

		
	def generatey0(self,i):
		if self.random==False:
			ix=int(i/(self.ny0))
			ip=i%(self.ny0)
			return np.array([-self.xmax/2.0+(ix+0.5)*self.dx,-self.pmax/2.0+(ip+0.5)*self.dp])
		else :
			return np.array([rd.randint(0,101)/100.0*self.xmax-self.xmax/2.0,rd.randint(0,101)/100.0*self.pmax-self.pmax/2.0])
			
			
			
class PhasePortrait:
	# This class can be used to generate a stroposcopic phase space of a 
	# given time periodic system. It requires time propagator, to include 
	# both discrete and continous time map.
	def __init__(self,nperiod,ny0,timepropagator,xmax=2*np.pi,pmax=2.0,random=False):
		self.nperiod=nperiod
		self.timepropagator=timepropagator
		
		self.xmax=xmax
		self.pmax=pmax
		
		self.ny0=ny0
		self.random=random
		self.np0=int(0.5*pmax/xmax*ny0)
		self.nx0=self.ny0-self.np0
		self.x0=np.linspace(0.0, self.xmax, num=self.nx0)
		self.p0=np.linspace(-self.pmax, self.pmax, num=self.np0)
	
	def getTrajectory(self,i):
		print(i)
		y=self.generatey0(i)
		x=np.zeros(self.nperiod)
		p=np.zeros(self.nperiod)
		for k in range(0,self.nperiod):
			y=self.timepropagator.propagate(y)
			x[k]=y[0]%(2*self.xmax)
			if x[k]>self.xmax:
				x[k]=x[k]-2*self.xmax
			p[k]=y[1]

		return x, p
		
	def getChaoticity(self,x,p):
		k=10
		H, xedges, yedges = np.histogram2d(x, p,bins=np.linspace(-np.pi,np.pi,k))
		s=0.0
		for i in range(0,k-1):
			for j in range(0,k-1):
				if H[i,j]>0.0:
					s+=1
		s/=(k-1)**2	
		return s
		
	
	def save(self,wdir=""):
		st=np.zeros(self.ny0)
		xt=np.zeros((self.ny0,self.nperiod))
		pt=np.zeros((self.ny0,self.nperiod))
		norms=0.0
		for i in range(0,self.ny0):
			xt[i,:],pt[i,:] = self.getTrajectory(i)
			st[i]=self.getChaoticity(xt[i,:],pt[i,:])
			norms=max(st[i],norms)
		st=st/norms
		np.savez(wdir+"trajectories","w", x=xt, p=pt,s=st)

	def npz2png(self,wdir=""):
		cmap=plt.get_cmap("jet")
		
		ax = plt.axes()
		ax.set_xlim(-self.xmax,self.xmax)
		ax.set_ylim(-self.pmax,self.pmax)
		ax.set_aspect('equal')
		
		data=np.load(wdir+"trajectories.npz")
		x=data["x"]
		p=data["p"]
		s=data["s"]
		
		for i in range(0,self.ny0):
			plt.scatter(x[i,:],p[i,:],s=0.50**2,c=cmap(s[i]))
		plt.savefig(wdir+"SPS.png")

		
	def generatey0(self,i):
		if self.random==False:
			if(i<self.nx0):
				return np.array([self.x0[i],0.0])
			else:
				return np.array([0.0,self.p0[int(i-self.nx0)]])		
		else :
			return np.array([rd.randint(0,101)/100.0*2*self.xmax-self.xmax,rd.randint(0,101)/100.0*2*self.pmax-self.pmax])
