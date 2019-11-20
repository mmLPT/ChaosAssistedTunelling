import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.systems.kickedrotor import *


# ~ gamma=0.375
# ~ e=0.242
# ~ h=1/2.88
# ~ x0=1.8

e=0.400
gamma=0.310
h=1/4.53
x0=1.8/4.5*np.pi

# ~ K=1.5
tmax=150

Ncell=5
ncellini=40
ncheck=5


# Simulation parameters
N=Ncell*32 # number of grid points
itmax=int(tmax/2) # number of period simulated (*2)
icheck=2

# ~ h=2*np.pi/N
print(N*h/Ncell)


SPSclassfile="tempdata/PP"

# Creation of objects
pot=PotentialMP(e,gamma)
# ~ pot=PotentialKR(K)
grid=Grid(N,h,xmax=Ncell*2*np.pi)
husimi=Husimi(grid)
fo=CATFloquetOperator(grid,pot)
fo.diagonalize()


		
		
np.savez("tempdata/2-Ncell-151-hm-4d53-N40",xm=xm,time=time,a=a,x=grid.x,b=b,p=p)		

	
	




