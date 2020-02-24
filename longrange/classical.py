import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.systems.kickedrotor import *



e=0.50
gamma=0.15
iperiod=50
ny0=150

pot=PotentialMP(e,gamma)
cp=ClassicalContinueTimePropagator(pot)
pp=PhasePortrait(iperiod,ny0,cp,xmax=np.pi,pmax=2.5) 
time=2*np.arange(iperiod)

x=np.zeros((ny0,iperiod))

x0=3*np.pi/4
# ~ x0=np.pi
# ~ x0=np.pi/4
p0=0.0

ax=plt.gca()
ax.set_ylabel(r"Moyenne de $\frac{|x-x_0|}{2\pi}$")
ax.set_xlabel(r"Number of periods")
for i in range(0,ny0):
	print(i)
	y0=np.array([np.random.normal(x0,0.01),np.random.normal(p0,0.01)])
	x[i]=pp.getOrbit(y0)[0]

xstd=np.std(x-x0,axis=0)/(2*np.pi)

fit = np.polyfit(time[25:-1],xstd[25:-1], 1)
		
v=fit[0]

print(v)



ax.plot(time,fit[0]*time+fit[1],c="red")
ax.scatter(time,xstd)
ax.set_ylim(0,np.max(xstd))
ax.set_xlim(0,np.max(time))	
ax.grid()

plt.show()

