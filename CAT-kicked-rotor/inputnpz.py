import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="input/K0d3"

# general information
N=24

# potential parameters
K=0.3

hmin=1/10.30
hmax=1/10.0

Kmin=0.0
Kmax=7.0

# free propagation/classical
iperiod=10000

np.savez(inputfile,"w", N=N,K=K,iperiod=iperiod,hmin=hmin,hmax=hmax,Kmin=Kmin,Kmax=Kmax)
