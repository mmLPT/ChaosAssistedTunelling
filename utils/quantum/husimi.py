import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from utils.quantum.grid import *
from utils.quantum.wavefunction import *
from utils import latex as latex

import matplotlib.image as mpimg

# This scripts contains: 1 class
# + class : Husimi

class Husimi:
	# The Husimi class provides a tool to generate Husimi representation
	# of wavefunctions. It is build from a grid, so you can generate 
	# representation of differents wave functions from a single object
	
	def __init__(self, grid, scale=1.0,pmax=2*np.pi):
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

	def getRho(self,wf,i1=0,i2=1):
		# Computes Husimi representation of a given wavefunction
		# It returns a 1-normalized 2D-density
		rho=np.zeros((self.Np,self.Nx))
		psip=np.fft.fftshift(wf.p)
		for ip in range(0,self.Np):	
			p0=self.p[ip]
			phi1=np.exp(-(self.pshift[ip]-p0)**2/(2*self.sigmap**2))
			for ix in range(int(i1*self.Nx),int(i2*self.Nx)):
				phi=phi1*np.exp(-(1j/self.h)*(self.x[ix]+self.xmax/2.0)*self.pshift[ip])
				rho[ip][ix]= abs(sum(np.conj(phi)*psip))**2
		nrm2=np.sum(rho)
		return rho/nrm2
		
	def save(self, wf, datafile,i1=0,i2=1):
		# Saves the Husimi representation in 'datafile.npz' file
		# If 'convert' is true, generates 'datafile.png' with npz2png
		rho=self.getRho(wf,i1=i1,i2=i2)
		np.savez(datafile,"w", rho=rho,x=self.x,p=self.p)
		
	def npz2png(self, datafile,SPSclassfile='',SPSclassbool=False,cmapl="Greys",textstr="",xmax=2*np.pi):
		# Get data from .npz file
		data=np.load(datafile+".npz")
		rho=data["rho"]
		x=data["x"]
		p=data["p"]
		R=rho.max()
		# ~ rho=rho/0.0040 #0.0016
		rho=rho/R
		
		
		fig = latex.fig(columnwidth=345.0,wf=1.0,hf=2.0/np.pi)
		
		# Generla settings : tile/axes
		ax=plt.gca()
		ax.set_ylim(-2,2)
		ax.set_xlim(-max(self.x),max(self.x))
		ax.set_aspect('equal')
		ax.set_xticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi])
		ax.set_xticklabels([r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
		ax.set_yticks([-0.5*np.pi,0,0.5*np.pi])
		ax.set_yticklabels([r"$-\pi/2$",r"$0$",r"$\pi/2$"])
		
		if SPSclassbool==True:
			img=mpimg.imread(SPSclassfile+".png")
			for i in range(0,int(max(self.x)/np.pi)+1):
				x1=min(self.x)+i*2*np.pi
				ax.imshow(img,extent=[x1,x1+2*np.pi,-2.0, 2.0])
			
		# 2D map options
		
		mnlvl=0.0
		levels = np.linspace(mnlvl,1.0,100)	
		cmap = plt.get_cmap(cmapl)
		norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
		plt.contourf(x,p,rho, levels=levels,cmap=cmap,alpha=0.5)
		levels = np.linspace(mnlvl,1.0,6)	
		# ~ levels = [0.10]
		# ~ contours=plt.contour(x,p,rho, levels=levels,colors="k",linestyles="-",linewidths=1.5)
		contours=plt.contour(x,p,rho, levels=levels,colors="k",linewidths=0.5)
		# ~ plt.clabel(contours, inline=True, fontsize=8)
		# ~ plt.pcolormesh(x,p,rho, norm=norm,cmap=cmap)
		
		
		# ~ dpx=np.sqrt(self.h)
		# ~ xticks=[0]
		# ~ x0=0
		# ~ while(x0+dpx<self.xmax/2):
			# ~ x0=x0+dpx
			# ~ xticks=np.append(xticks,x0)
			# ~ xticks=np.append(xticks,-x0)
			
		# ~ pticks=[0]
		# ~ p0=0
		# ~ while(p0+dpx<2):	
			# ~ p0=p0+dpx
			# ~ pticks=np.append(pticks,p0)
			# ~ pticks=np.append(pticks,-p0)

			
		# ~ ax.set_xticks(xticks,minor=True)
		# ~ ax.set_yticks(pticks,minor=True)
		# ~ ax.grid(which='minor', color="black",alpha=0.2)
		
		
		# ~ props = dict(boxstyle='square', facecolor='white', alpha=1.0)

		# ~ ax.text(0.50, 0.90, textstr, transform=ax.transAxes, fontsize=14,va='center',ha='center', bbox=props)
		
		ax.set_title(textstr)
		
		# ~ fig.set_frameon(False)
		# ~ fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
		# ~ ax.set_xticks([])
		# ~ ax.set_yticks([])
		# ~ ax.set_xticklabels([])
		# ~ ax.set_yticklabels([])

		# Saving fig
		latex.save(datafile,form="png",dpi=500,transparent=False,bbox_inches='')
		plt.close(fig)
	
