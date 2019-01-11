import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt
import modesbasic
import modesxconfinment
import modesquasimomentum

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
import utils.plot.read as read

N=128
h=0.1 #0.25
e=0.300
gamma=0.295
grid=Grid(N,h)

#f=PeriodicFunctions(phase=np.pi)
pot=PotentialMP(e,gamma)

compute=True
read=True

#~ modesbasic.classical(pot,nperiod=100,ny0=25,wdir="classical-trajectories/",compute=compute)
#~ modesbasic.period_with_h(e=0.29, gamma=0.29, imax=50, N=128, datafile="split")
#~ modesbasic.explore_epsilon_gamma()
#~ modesbasic.propagate( grid, pot, iperiod=300, icheck=10,wdir="propagation-test/",projfile="projs",husimibool=True)

#~ modesxconfinment.perturbation_theory(grid,e,gamma,compute=compute=,read=read)
#~ modesxconfinment.track_crossing(N=64, e=0.315, gamma=0.290, hmin=0.298, hmax=0.302,datafile="data/track_crossing4")
#~ modesxconfinment.check_T_with_confinment(imax=220,e=e,gamma=gamma)
#~ modesxconfinment.symetry_of_gs_with_h(N=64, e=0.315, gamma=0.290, datafile="data/croisement3",compute=compute,read=read)

modesquasimomentum.imaginary()




