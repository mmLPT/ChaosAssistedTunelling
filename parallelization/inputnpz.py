import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="data/input/s-28-84-nu-80"

# general information
N=64
description="around 1st oscillation"

# potential parameters
e=0.44
x0=1.50
#gamma=0.35
#h=0.250
s=28.84
nu=80.0*10**3
gamma, h = convert2theory(s=s, nu=nu)

# quasi-momentum
beta0=0.0*h
Ndbeta=6.0
dbeta=h/(3.0*Ndbeta)

# spectrum
hmin=0.15
hmax=0.45
nstates=30

# free propagation/classical
iperiod=3000

np.savez(inputfile,"w", description=description, N=N, e=e, gamma=gamma, h=h, x0=x0, beta0=beta0, dbeta=dbeta, Ndbeta=Ndbeta, iperiod=iperiod,hmin=hmin,hmax=hmax,nstates=nstates)
