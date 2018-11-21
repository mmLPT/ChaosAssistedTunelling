import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import time

from utils.quantum.grid import *
from utils.quantum.wavefunction import *

# The following class computes 

class Husimi:
	def __init__(self, grid, scale=3.0,xmax=0,pmax=2*np.pi):
		self.grid=grid
		# husimi grid
		self.scale=scale # scale for husimi grid
		self.h=grid.h
		self.N=grid.N
		
		#keep aspect ratio
		self.sigmap=np.sqrt(self.h/2.0)
		self.sigmax=np.sqrt(self.h/2.0)
		
		# Maximum values to plot. You often doesn't need high p representation
		self.xmax=xmax
		if self.xmax==0:
			self.xmax=grid.xmax
		self.pmax=pmax
		
		# The husimi grid
		self.Nx=int(self.xmax/self.sigmax*self.scale)
		self.x=np.linspace(-self.xmax/2.0,self.xmax/2.0,self.Nx)
		self.Np=int(self.pmax/self.sigmap*self.scale)
		self.p=np.linspace(-self.pmax/2.0,self.pmax/2.0,self.Np)
		
		
		self.pshift=np.zeros((self.Np,self.N),dtype=np.complex_)
		pgrid=np.fft.fftshift(grid.p)
		for ip in range(0,self.Np):
			i0=int((self.p[ip]+self.N*self.h/2.0)/self.h)
			for i in range(0,self.N):
				self.pshift[ip][i]=pgrid[i]
				if i< i0-self.N/2:
					self.pshift[ip][i]=pgrid[i]+self.N*self.h
				if i> i0+self.N/2:
					self.pshift[ip][i]=pgrid[i]-self.N*self.h	

	def getRho(self,wf):
		rho=np.zeros((self.Np,self.Nx))
		psip=np.fft.fftshift(wf.p)
		for ip in range(0,self.Np):	
			p0=self.p[ip]
			phi1=np.exp(-(self.pshift[ip]-p0)**2/(2*self.sigmap**2))
			for ix in range(0,self.Nx):
				phi=phi1*np.exp(-(1j/self.h)*(self.x[ix]+self.xmax/2.0)*self.pshift[ip])
				rho[ip][ix]= abs(sum(np.conj(phi)*psip))**2
		nrm2=np.sum(rho)
		return rho/nrm2
		
	def save(self, wf, datafile, title="", convert=True):
		rho=self.getRho(wf)
		np.savez(datafile,"w", rho=rho,x=self.x,p=self.p)
		if convert:
			self.npz2png(datafile, title=title)
		
	def npz2png(self, datafile, title=""):
		data=np.load(datafile+".npz")
		rho=data["rho"]
		x=data["x"]
		p=data["p"]
		R=rho.max()
		
		plt.title(title)
				
		levels = MaxNLocator(nbins=100).tick_values(0.0,R)	
		cmap = plt.get_cmap('Spectral')
		norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
		
		ax = plt.axes()
		ax.set_ylim(-self.pmax/2.0,self.pmax/2.0)
		ax.set_xlim(-self.grid.xmax/2.0,self.grid.xmax/2.0)
		ax.set_aspect('equal')
		"""
		ax.set_xticks([-pi, -pi/2.0,0, pi/2.0, pi])
		ax.set_xticklabels([r"$-\pi$",r"$-\frac{\pi}{2}$","$0$", r"$\frac{\pi}{2}$",r"$\pi$"])
		
		"""
		ax.set_yticks([-np.pi, -np.pi/2.0,0, np.pi/2.0, np.pi])
		ax.set_yticklabels([r"$-\pi$",r"$-\frac{\pi}{2}$","$0$", r"$\frac{\pi}{2}$",r"$\pi$"])

		#plt.contourf(x,p,rho, levels=levels,cmap=cmap)
		plt.pcolormesh(x,p,rho, norm=norm,cmap=cmap)
		plt.savefig(datafile+".png", bbox_inches='tight')
		plt.clf() 
