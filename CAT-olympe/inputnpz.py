import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="input/e_0d15-g_0d25"

# general information
N=64
description="interactions"

# potential parameters
e=0.15
gamma=0.25
h=0.4
x0=0.8
s=9.75
nu=52.68*10**3
x0exp=28.6
#gamma, h, x0 = convert2theory(s=s, nu=nu,x0exp=x0exp)

# quasi-momentum
beta0=0.0*h
Ndbeta=3.0

# spectrum
nstates=4

# explore
hmin=0.4
hmax=0.7
emin=0.42
emax=0.44
gammamin=0.26
gammamax=0.32
gmin=0.0
gmax=1.0

# free propagation/classical
iperiod=3000

np.savez(inputfile,"w", description=description, N=N, e=e, gamma=gamma, h=h, x0=x0, beta0=beta0, Ndbeta=Ndbeta, iperiod=iperiod,hmin=hmin,hmax=hmax,nstates=nstates,emin=emin,emax=emax,gammamin=gammamin,gammamax=gammamax,gmin=gmin,gmax=gmax)
