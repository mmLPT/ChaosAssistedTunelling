import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# DIRTY UNCOMMENTED FUNCTIONS, mostly used to plot data

def latex():
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

def wf(datafile=""):
	data=np.load(datafile+".npz")
	x=data["x"]
	psix=np.real(data["psix"])
	ax = plt.axes()
	ax.set_xlim(min(x),max(x))
	ax.set_ylim(-1.1*max(psix),1.1*max(psix))
	plt.plot(x,psix,c="blue")
	plt.show()

def wf_special(datafile="",N=0,xmax=0,g1=0.0):
	data=np.load(datafile+".npz")
	x=data["x"]
	p=data["x"]
	psix=data["psix"]
	psip=data["psip"]
	ax = plt.axes()
	ax.set_xlim(min(x),max(x))
	ax.set_ylim(0,1.1*max(abs(psix)**2))
	plt.plot(x,abs(psix)**2,c="blue")
	omega=1.0
	h=0.1
	mxpsi=max(abs(psix))**2
	
	
	Rexp=3.0/(4.0*mxpsi)*((xmax/N)**0.5)**2
	gexp=2.0*omega**2*Rexp**3/3.0
	Rth=(3*g1/(2*omega**2))**(1/3.0)
	R=Rth
	
	#phi=np.exp(-0.5*omega*x**2/h)*(omega/(np.pi*h))**0.25*(5*2*np.pi/N)**0.5
	#phi=np.sqrt(3.0/(4.0*R))*np.sqrt(1-(x/R)**2)*(5*2*np.pi/N)**0.5
	
	phi=np.zeros(N)
	for i in range(0,N):
		if abs(x[i])>R:
			phi[i]=0
		else:
			phi[i]=np.sqrt(3.0/(4.0*R))*np.sqrt(1-(x[i]/R)**2)*(xmax/N)**0.5

	plt.plot(x,phi**2,c="red")
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
	plt.xlim(min(data['h']),max(data['h']))
	h=data['h']
	qE=abs(data['qE'][:,1]-data['qE'][:,0])
	T=data['T']
	
	plt.yscale('log')
	plt.ylim(1,100000)
	plt.plot(h,T)
	plt.show()
		
	plt.yscale('log')
	#plt.xlim(min(h),max(h))
	#plt.ylim(0.000001,1)
	plt.plot(1.0/h,qE)
	plt.show()
	
def mode5(datafile="split"):
	data=np.load(datafile+".npz")
	#plt.xlim(min(data['h']),max(data['h']))
	h=data['h']
	T0=data['T0']
	Tasym=data['Tasym']
	isLowerSymetric=data['isLowerSymetric']
	
	N=h.shape[0]
	Tbool=np.zeros(N,dtype=bool)
	for i in range(0,N):
		Tbool[i]=(T0[i]>Tasym[i])
		
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	
	plt.xlim(min(data['h']),max(data['h']))
	
	ax1.scatter(h,isLowerSymetric)
	ax1.scatter(h,Tbool)
	ax2.set_yscale("log")
	ax2.plot(h,T0)
	
	plt.show()
	
def projs(datafile):
	data=np.load(datafile+".npz")
	time=data['time']
	projs=data['projs']

	plt.ylim(1.1*min(projs),1.1*max(projs))
	plt.xlim(0,max(time))
	plt.plot(time,projs,c="blue")
	#plt.plot(time,1-projs,c="red")
	#plt.show()
	plt.savefig(datafile+".png", bbox_inches='tight')
	plt.clf() 
	
def asym(datafile):
	latex()
	
	# loading file
	data=np.load(datafile+".npz")
	x=data['x1']
	T=data['T']
	Tth=data['Tth']
	qE01=data['qE0'][0]
	qE02=data['qE0'][1]
	qE1=data['qE'][:,0]#-qE01
	qE2=data['qE'][:,1]#-qE02
	qEth1=data['qEth'][:,0]#-qE01
	qEth2=data['qEth'][:,1]#-qE02

	plt.xlim(-max(x),max(x))
	plt.xlabel(r'Harmonic potential shift')
	plt.ylabel(r'Quasi-energies')
	
	plt.plot(x,qE1,c='red')
	plt.plot(x,qE2,c='blue')
	plt.scatter(x,qEth1,c='orange',s=4.0**2)
	plt.scatter(x,qEth2,c='green',s=4.0**2)
	plt.show()

	plt.xlim(-max(x),max(x))
	plt.plot(x,T,c='red')
	plt.scatter(x,Tth,c='orange',s=4.0**2)
	plt.show()
	
	
def splitWithSym(datafile="split",savefig=False):
	data=np.load(datafile+".npz")

	
	isLowerSymetric=data['isLowerSymetric']
	h=data['h']
	#dqE=data['dqE']
	qE0=data['qE'][:,0]
	qE1=data['qE'][:,1]
	T=data['T']
	
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	
	
	plt.xlim(min(data['h']),max(data['h']))
	#plt.yscale('log')
	"""
	ax1.scatter(h,qE0)
	ax1.scatter(h,qE1)
	ax2.scatter(h,isLowerSymetric)
	plt.show()
	"""
	ax1.set_yscale("log")
	ax1.plot(h,T)
	ax2.scatter(h,isLowerSymetric)
	plt.show()

	
	




