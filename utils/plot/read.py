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

	
	




