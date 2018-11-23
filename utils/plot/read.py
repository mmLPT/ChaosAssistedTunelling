import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt

# DIRTY UNCOMMENTED FUNCTIONS, mostly used to plot data

def wf(datafile=""):
	data=np.load(datafile+".npz")
	x=data["x"]
	p=data["x"]
	psix=data["psix"]
	psip=data["psip"]
	ax = plt.axes()
	ax.set_xlim(min(x),max(x))
	ax.set_ylim(0,1.1*max(abs(psix)**2))
	plt.plot(x,abs(psix)**2)
	omega=2.0
	h=0.1
	N=2048
	g=1500.0
	R=(3*g/(2*omega**2))**(1/3.0)
	#phi=np.exp(-0.5*omega*x**2/h)*(omega/(np.pi*h))**0.25*(5*2*np.pi/N)**0.5
	phi=np.sqrt(3.0/(4.0*R))*np.sqrt(1-(x/R)**2)*(5*2*np.pi/N)**0.5
	phi=np.zeros(N)
	for i in range(0,N):
		if abs(x[i])>R:
			phi[i]=0
		else:
			phi[i]=np.sqrt(3.0/(4.0*R))*np.sqrt(1-(x[i]/R)**2)*(5*2*np.pi/N)**0.5
	#phi=0
	print(sum(phi**2))
	plt.plot(x,phi**2)
	plt.show()
	"""plt.savefig(datafile+"x.png", bbox_inches='tight')
	plt.clf() 
	ax = plt.axes()
	ax.set_xlim(min(p),max(p))
	ax.set_ylim(0,1.1*max(abs(psip)**2))
	plt.plot(p,abs(psip)**2)
	plt.savefig(datafile+"p.png", bbox_inches='tight')
	plt.clf() """

def split(datafile="split",savefig=False):
	data=np.load(datafile+".npz")
	plt.yscale('log')
	plt.xlim(min(data['h']),max(data['h']))
	plt.ylim(1,100000)
	plt.plot(data['h'],data['T'])
	if savefig:
		plt.savefig(datafile+"-T.png", bbox_inches='tight')
		plt.clf() 
	else:
		plt.show()
		
	plt.yscale('log')
	plt.xlim(min(1/data['h']),max(1/data['h']))
	plt.ylim(0.000001,1)
	plt.plot(1/data['h'],data['qE'])
	if savefig:
		plt.savefig(datafile+"-qE.png", bbox_inches='tight')
		plt.clf() 
	else:
		plt.show()
	
def projs(datafile):
	data=np.load(datafile+".npz")
	time=data['time']
	projs=data['projs']

	plt.ylim(0,1.0)
	plt.xlim(0,max(time))
	plt.plot(time,projs,c="blue")
	plt.plot(time,1-projs,c="red")
	
	plt.savefig(datafile+".png", bbox_inches='tight')
	plt.clf() 
	
def asym(datafile):
	data=np.load(datafile+".npz")
	x=data['x1']
	T=data['T']
	Tth=data['Tth']
	plt.ylim(153.306663132,153.306954537)
	plt.xlim(-max(x),max(x))
	plt.scatter(x,T,c='red')
	plt.scatter(x,2*Tth[5]-Tth,c='blue')
	#plt.plot(data['x1'],data['Tth'],c='red')
	plt.show()
	
	




