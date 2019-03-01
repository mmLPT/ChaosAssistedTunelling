import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="input/10-moyen"

# general information
N=64
description="10ieme data set 12/2018"

# potential parameters
#x0=1.50
#gamma=0.35
#h=0.250
e=0.44
s=13.75
nu=55.6*10**3
x0exp=90.0
gamma, h,x0 = convert2theory(s=s, nu=nu,x0exp=x0exp)

# quasi-momentum
beta0=0.0*h
Ndbeta=2.0
dbeta=h/(3.0*Ndbeta)

# spectrum
hmin=0.15
hmax=0.45
nstates=30

# free propagation/classical
iperiod=10000

np.savez(inputfile,"w", description=description, N=N, e=e, gamma=gamma, h=h, x0=x0, beta0=beta0, dbeta=dbeta, Ndbeta=Ndbeta, iperiod=iperiod,hmin=hmin,hmax=hmax,nstates=nstates)
