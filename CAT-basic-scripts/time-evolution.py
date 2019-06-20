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
s=28.2
nu=81.11*10**3
x0exp=20.0
e=0.15
gamma, h, x0 = convert2theory(s=s, nu=nu,x0exp=x0exp)

tmax=200

print(convert2exp(gamma,h,x0))
print(gamma,h,x0)

print(28.6*1.5)
#x0=x0*1.5

#gamma=0.230
#e=0.418
#h=0.292
#x0=0.9*1.5

# Simulation parameters
N=64 # number of grid points
itmax=int(tmax/2) # number of period simulated (*2)
icheck=1
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
husimi=Husimi(grid)

### SIMULATION ###

# 1 - Creation of the initial state
wf.setState("coherent",x0=-x0,xratio=2.0)

# 2 - Propagation
for it in range(0,itmax):
	fo.propagate(wf)
	xL[it]=wf.getxM(-np.pi,-xm)
	xR[it]=wf.getxM(xm,np.pi)
	xM[it]=wf.getxM(-xm,xm)
	#~ if it%icheck==0:
		#~ print(it)
		#~ husimi.save(wf,"data/hu-"+strint(int(it/icheck)*2))
		

# 3 - Normalization
norm=xR[0]+xL[0]+xM[0]
xL=xL/norm
xR=xR/norm
xM=xM/norm

# 4 - Output
if savenpz:
	np.savez(datafile,xL=xL,xR=xR,xM=xM,time=time,gamma=gamma,e=e,x0=x0,h=h)
if plot:
	
	plt.plot(time,xL)
	plt.plot(time,xR)
	plt.plot(time,xM)
	ax=plt.gca()
	s,nu,x0exp=convert2exp(gamma,h,x0)
	ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f} \quad h={:.3f} \quad x0={:.1f}$".format(e,gamma,h,x0)+"\n"+r"$s={:.3f} \quad \nu={:.3f} kHz \quad x_0={:.1f}^o$".format(s,nu/10**3,x0exp))
	ax.set_xlim(0,tmax)
	ax.set_ylim(0.0,1.0)
	plt.show()


