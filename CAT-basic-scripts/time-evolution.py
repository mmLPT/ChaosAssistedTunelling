import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *

# Physical parameters
e=0.44
gamma=0.267
h=0.292
x0=np.pi/2 # initial position of wavefunction

# Simulation parameters
N=64 # number of grid points
itmax=25 # number of period simulated (*2)
xm=0.0 # if you want a thrid observable centered between - and + xm

# output 
plot=True
savenpz=True
datafile="data"

# Creation of objects
pot=PotentialMP(e,gamma)
grid=Grid(N,h)
husimi=Husimi(grid)
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)
xL=np.zeros(itmax)
xR=np.zeros(itmax)
xM=np.zeros(itmax)
time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)

### SIMULATION ###

# 1 - Creation of the initial state
wf.setState("coherent",x0=x0,xratio=2.0)

# 2 - Propagation
for it in range(0,itmax):
	fo.propagate(wf)
	xL[it]=wf.getxM(-np.pi,-xm)
	xR[it]=wf.getxM(xm,np.pi)
	xM[it]=wf.getxM(-xm,xm)

# 3 - Normalization
norm=xR[0]+xL[0]+xM[0]
xL=xL/norm
xR=xR/norm
xM=xM/norm

# 4 - Output
if savenpz:
	np.savez(datafile,xL=xL,xR=xR,xM=xM,time=time)
if plot:
	plt.plot(time,xL)
	plt.plot(time,xR)
	plt.plot(time,xM)
	plt.show()


