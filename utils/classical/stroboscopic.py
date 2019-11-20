import matplotlib.pyplot as plt
import numpy as np
import random as rd
from utils.classical.timepropagator import *
from utils.toolsbox import *
					
class PhasePortrait:
	# This class can be used to generate a stroposcopic phase space of a 
	# given time periodic system. It requires time propagator, to include 
	# both discrete and continous time map.
	def __init__(self,nperiod,ny0,timepropagator,xmax=np.pi,pmax=np.pi,random=False,theta1=0,theta2=0,X0=0,Y0=0):
		self.nperiod=nperiod
		self.timepropagator=timepropagator
		
		self.xmax=xmax
		self.pmax=pmax
		self.theta1=theta1
		self.theta2=theta2
		
		self.ny0=ny0
		self.random=random
		self.np0=int(0.5*pmax/(xmax+pmax)*ny0)
		self.nx0=self.ny0-self.np0
		self.x0=np.linspace(-self.xmax, self.xmax, num=self.nx0)
		self.p0=np.linspace(-self.pmax, self.pmax, num=self.np0)
		
		# ~ def r(theta0,r1M=np.pi,r2M=1):
			# ~ R=r2M/(np.abs(np.sin(theta0))+r2M/r1M*np.abs(np.cos(theta0)))
			# ~ return np.array([R*np.cos(theta0),R*np.sin(theta0)])
			
		def r(theta0,r1M=np.pi,r2M=1):
			b=r2M
			a=r1M
			print(a!=0)
			e=np.sqrt(a**2-b**2)/a
			R=b/np.sqrt(1-e**2*np.cos(theta0)**2)
			return np.array([R*np.cos(theta0),R*np.sin(theta0)])
		
		r1max=np.pi
		r1min=-r1max
		r2max=1
		r2min=-r2max
		r3max=1.5*self.pmax
		r3min=Y0
		r4max=-r3max
		r4min=-r3min
		
		rtheta1norm=2*np.sqrt(np.sum(r(theta1,r1M=r1max,r2M=r2max)**2))
		rtheta2norm=2*np.sqrt(np.sum(r(theta2,r1M=r1max,r2M=r2max)**2))
		rtheta3norm=np.abs(r3max-r3min)
		rtheta4norm=np.abs(r4max-r4min)
		normtot=rtheta1norm+rtheta2norm+rtheta3norm+rtheta4norm
		
		n2=int(rtheta2norm*ny0/normtot)
		n3=int(rtheta3norm*ny0/normtot)
		n4=int(rtheta4norm*ny0/normtot)
		n1=ny0-n4-n3-n2

		rtheta1=r(theta1,r1M=np.linspace(r1min, r1max, num=n1),r2M=np.linspace(r2min, r2max, num=n1))
		rtheta2=r(theta2,r1M=np.linspace(r1min, r1max, num=n2),r2M=np.linspace(r2min, r2max, num=n2))
		rtheta3=np.linspace(r3min, r3max, num=n3)
		rtheta4=np.linspace(r4min, r4max, num=n4)
		
		a=np.concatenate((rtheta1[0],rtheta2[0],X0*np.ones(n3),-X0*np.ones(n4)))
		b=np.concatenate((rtheta1[1],rtheta2[1],rtheta3,rtheta4))
		self.y0=np.zeros((ny0,2))
		self.y0[:,0]=a
		self.y0[:,1]=b
		
	
	def getTrajectory(self,i):
		y=self.generatey0(i)
		x=np.zeros(self.nperiod)
		p=np.zeros(self.nperiod)
		for k in range(0,self.nperiod):
			x[k]=y[0]%(2*self.xmax)
			if x[k]>self.xmax:
				x[k]=x[k]-2*self.xmax
			p[k]=y[1]
			y=self.timepropagator.propagate(y)

		return x, p
		
	def getOrbit(self,y0):
		y=y0
		x=np.zeros(self.nperiod)
		p=np.zeros(self.nperiod)
		for i in range(0,self.nperiod):
			y=self.timepropagator.propagate(y)
			y=self.timepropagator.propagate(y)
			x[i]=y[0]
			p[i]=y[1]
		return x,p
		
	def getChaoticity(self,x,p):
		k=20
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
			# ~ if(i<self.nx0):
				# ~ return np.array([self.x0[i],0.0])
			# ~ else:
				# ~ return np.array([0.0,self.p0[int(i-self.nx0)]])
				
			return self.y0[i]
		else :
			return np.array([rd.randint(0,101)/100.0*2*self.xmax-self.xmax,rd.randint(0,101)/100.0*2*self.pmax-self.pmax])
