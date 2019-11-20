import sys
sys.path.insert(0, '..')
import numpy as np
# ~ from utils.toolsbox import *

inputfile="input/e0d05-g0d15"

# general information
Ncell=201
N=2*64
description=""

# potential parameters
e=0.05
gamma=0.15
h=1/1.66
x0=0.0
# explore
hmin=1/10
hmax=1/1

# free propagation/classical
iperiod=100

np.savez(inputfile,"w", description=description, Ncell=Ncell, e=e, gamma=gamma, h=h, x0=x0, iperiod=iperiod,hmin=hmin,hmax=hmax,N=N)
