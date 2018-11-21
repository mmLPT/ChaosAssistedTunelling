import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

#from utils.quantum import *
#from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
#from utils.mathtools.periodicfunctions import *
import utils.plot.read as read
	
	
N=128+32+16+4+2+1+1
gamma=0.29
e=0.35
h=0.1

grid=Grid(N,h)
pot=PotentialMP(e,gamma)		


#propagate(grid, pot, 1000, 50,"testfunc/",T0=4*np.pi,idtmax=500)
#classical(pot,nperiod=50,ny0=300,"testfunc/")
#splitting_with_h(N, ht=np.linspace(1/10.0,1/7.5,250), e, gamma, wdir="testfunc/")


#explore_e_gamma()
wdir="asym/"
datafile="data"+str(int(N))
explore_asymetry(grid,e,gamma,wdir=wdir,datafile=datafile)
read.asym(wdir+datafile)

#read.split("testfunc/split")



