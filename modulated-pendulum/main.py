import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
#from utils.mathtools.periodicfunctions import *
import utils.plot.read as read
	
	
N=2048*10 #128+32+16+4+2+1+1
gamma=0.29
e=0.35
h=0.1

grid=Grid(N,h,xmax=20*2*np.pi)
pot=PotentialMP(e,gamma)		


#propagate(grid, pot, 1000, 50,"testfunc/",T0=4*np.pi,idtmax=500)

#classical(pot,nperiod=50,ny0=300,"testfunc/")

#splitting_with_h(N, ht=np.linspace(1/10.0,1/7.5,250), e, gamma, wdir="testfunc/")
#read.split("testfunc/split")

#explore_e_gamma()

#explore_asymetry(grid,e,gamma,wdir="asym/",datafile="data"+str(int(N)))
#read.asym("asym/data"+str(int(N)))
pot=PotentialMPasym(e,gamma,0,h)	
qitp=QuantumImaginaryTimePropagator(grid,pot,T0=4*np.pi,idtmax=3500)
wf=WaveFunction(grid)
husimi=Husimi(grid)
wf.setState("diracp")
wf=3.0*wf
wf.normalize()
print(wf//wf)
#husimi.save(wf,"gs")
wf.save("wf0")
read.wf("wf0")
wf=qitp.getGroundState(wf)
wf.save("gs")
read.wf("gs")




