import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from utils.quantum.grid import *
from utils.quantum.wavefunction import *
from utils.plot.latex import *

# This scripts contains: 1 class
# + class : Husimi

class Husimi:
	# The Husimi class provides a tool to generate Husimi representation
	# of wavefunctions. It is build from a grid, so you can generate 
	# representation of differents wave functions from a single object
	
	def __init__(self, grid, scale=3.0,pmax=2*np.pi):
		self.grid=grid
		self.scale=scale 
		# Husimi grid is defined over coherent states, but you can  
		# fix an higher resolution by changing 'scale'
		self.h=grid.h
		self.N=grid.N
		
		# Sigma of a coherent state with 1:1 aspect ratio
		self.sigmap=np.sqrt(self.h/2.0)
		self.sigmax=np.sqrt(self.h/2.0)
		
		# Boundaries for plotting
		self.xmax=grid.xmax
		self.pmax=pmax
		
		# Building Husimi grid
		self.Nx=int(self.xmax/self.sigmax*self.scale)
		self.x=np.linspace(-self.xmax/2.0,self.xmax/2.0,self.Nx)
		self.Np=int(self.pmax/self.sigmap*self.scale)
		self.p=np.linspace(-self.pmax/2.0,self.pmax/2.0,self.Np)
		
		# The following is a trick : as we are working on a periodic 
		# grid, we want
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
		# Computes Husimi representation of a given wavefunction
		# It returns a 1-normalized 2D-density
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
		# Saves the Husimi representation in 'datafile.npz' file
		# If 'convert' is true, generates 'datafile.png' with npz2png
		rho=self.getRho(wf)
		np.savez(datafile,"w", rho=rho,x=self.x,p=self.p)
		if convert:
			self.npz2png(datafile, title=title)
		
	def npz2png(self, datafile, title=""):
		# Converts an .npz file to .png file
		setLatex()
		# Get data from .npz file
		data=np.load(datafile+".npz")
		rho=data["rho"]
		x=data["x"]
		p=data["p"]
		R=rho.max()
			
		# Generla settings : tile/axes
		plt.title(title)
		ax = plt.axes()
		#ax.set_ylim(-self.pmax/2.0,self.pmax/2.0)
		ax.set_ylim(-1.0,1.0)
		ax.set_xlim(-self.grid.xmax/2.0,self.grid.xmax/2.0)
		ax.set_aspect('equal')
		"""
		ax.set_xticks([-pi, -pi/2.0,0, pi/2.0, pi])
		ax.set_xticklabels([r"$-\pi$",r"$-\frac{\pi}{2}$","$0$", r"$\frac{\pi}{2}$",r"$\pi$"])
		ax.set_yticks([-np.pi, -np.pi/2.0,0, np.pi/2.0, np.pi])
		ax.set_yticklabels([r"$-\pi$",r"$-\frac{\pi}{2}$","$0$", r"$\frac{\pi}{2}$",r"$\pi$"])
		"""
		
		
		# 2D map options
		levels = MaxNLocator(nbins=100).tick_values(0.0,R)	
		cmap = plt.get_cmap('binary')
		norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
		plt.contourf(x,p,rho, levels=levels,cmap=cmap)
		#plt.pcolormesh(x,p,rho, norm=norm,cmap=cmap)
		
		# Saving fig
		plt.savefig(datafile+".png", bbox_inches='tight')
		plt.clf() 
