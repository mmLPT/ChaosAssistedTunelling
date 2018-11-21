import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt

# DIRTY


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
	plt.savefig(datafile+".png", bbox_inches='tight')
	plt.clf() 

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
	#plt.ylim(105.829,105.83)
	plt.plot(data['x1'],data['T']**2,c='blue')
	#plt.plot(data['x1'],data['Tth'],c='red')
	plt.show()
	
	




