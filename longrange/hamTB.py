import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from scipy.linalg import expm, sinm, cosm
from utils.mathtools.periodicfunctions import *
import matplotlib as mpl

# ~ from hamTB import *

# ==================================================================== #
# Ce script permet de calculer le spectre de l'opérateur de Floquet en 
# fonction du quasimoment beta
# Il permet ensuite de reconstruire un Hamiltonien effectif
# ==================================================================== 


class TightBinding:
	def __init__(self,wdir,Ncell=0,F=0):
		self.Ncell=Ncell
		
		data=np.load(wdir+"gathered.npz")
		self.h=data['h']
		self.beta=data['beta']
		self.qEs=data['qEs']
		evecx=data['evecx']
		evecp=data['evecp']
		Npcell=data['evecp'].shape[1]
		data.close()
		
		Ncellmax=self.qEs[:,-1].size
		
		if Ncell==0:
			self.Ncell=Ncellmax
			
			
		
			
		ind=np.arange(0,Ncellmax,int(Ncellmax/self.Ncell),dtype=int)+int(0.5*Ncellmax/self.Ncell)
		
		self.qEs=self.qEs[ind,-1]
		self.beta=self.beta[ind,-1]/self.h*2*np.pi
		# ~ self.qEs=np.roll(self.qEs,int((self.qEs.size+1)/2))
		# ~ self.beta=np.roll(self.beta,int((self.beta.size+1)/2))
		
		print(self.Ncell,self.qEs.size)
		
		# Detérmination des éléments de couplage TB
		self.Vn=np.fft.rfft(self.qEs)/self.qEs.size
		
		print(self.Vn.size)
		
		
		# Construction de l'Hamiltonien TB
		self.H=np.zeros((self.Ncell,self.Ncell),dtype=complex)

		for i in range(0,self.Ncell):
			for j in range(i,self.Ncell):
				if np.abs(i-j)<=int(0.5*(self.Ncell-1)):
					self.H[i,j]=self.Vn[np.abs(i-j)]
				else:
					self.H[i,j]=np.conj(self.Vn[self.Ncell-np.abs(i-j)])
				self.H[j,i]=np.conj(self.H[i,j])
				
		i0=int(0.5*(self.Ncell-1))
		for i in range(0,self.Ncell):
			self.H[i,i]=self.H[i,i]+F*(i-i0)
		
		self.U=expm(-1j*self.H*4*np.pi/self.h)
		self.wf=np.zeros(self.Ncell,dtype=complex)
		
		
		
		#### WF0
		grid=Grid(self.Ncell*Npcell,self.h,xmax=self.Ncell*2*np.pi)
		self.wf0=WaveFunction(grid)
		ind0=np.flipud(np.arange(0,self.Ncell*Npcell,self.Ncell))

		for i in range(0,self.Ncell):
			self.wf0.p[ind0]=np.abs(evecp[ind[i]])
			ind0=ind0+1
			
		self.wf0.p=np.roll(self.wf0.p,int(self.Ncell/2+1))
		self.wf0.p=np.fft.fftshift(self.wf0.p*grid.phaseshift)
			
		self.wf0.p2x()
		self.wf0.normalizeX()
		self.wf0.x2p()
		
		
	def autocorrelation(self):
		gamma=np.zeros(self.Ncell)
		
		for i in range(0,self.Ncell):
			gamma[i]=np.trapz(np.multiply(self.qEs,np.roll(self.qEs,i)),x=self.beta)/(2*np.pi)	
		return gamma
		
		
		
		
	def propagate(self):
		self.wf=np.matmul(self.U,self.wf)
		
	def xstd(self):
		ind=np.arange(self.Ncell)
		return np.sqrt(np.sum((ind-self.xmean())**2*np.abs(self.wf)**2))
		
	def xmean(self):
		ind=np.arange(self.Ncell)
		return np.sum(ind*np.abs(self.wf)**2)
		
	def Vnth(self):
		diff=np.abs(self.qEs-np.roll(self.qEs,1))
		
		ind= ((np.diff(np.sign(np.diff(diff))) < 0).nonzero()[0]+1)
		
		beta0=self.beta[ind]
		W=diff[ind]
		
		ind2=beta0>0
		beta0=beta0[ind2]
		W=W[ind2]
		
		beta0=beta0-0.5*(self.beta[1]-self.beta[0])
		
		
		
		ax=plt.subplot(1,3,1)
		
		ax.plot(self.beta,self.qEs)
		
		ax=plt.subplot(1,3,2)
		ax.plot(self.beta,diff)
		ax.scatter(beta0,W)
		
		n=np.arange(self.Vn.size)
		
		
		def Vnth2(n,W,beta0):
			return -W/(2*np.pi*n)*2*np.sin(beta0*n)
		
		ax=plt.subplot(1,3,3)
		
		ax=plt.gca()
		ax.set_yscale('log')
		ax.set_xlim(0,200)
		# ~ ax.set_ylim(10**(-8),10**(-3))
		
		Vn=np.delete(self.Vn,0)
		
		Vnth0=np.zeros(n.size)
	
		for i in range(0,W.size):
			Vnth0+=Vnth2(n,W[i],beta0[i])
		
		Vnth0=np.delete(Vnth0,0)
		
		plt.scatter(np.arange(Vn.size),np.abs(Vn))
		plt.scatter(np.arange(Vn.size),np.abs(Vnth0))
		
		
		
		plt.show()
	
			
		
		
