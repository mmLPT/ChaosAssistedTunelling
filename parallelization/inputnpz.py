import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="input/regular"

# general information
N=64
description=""

# potential parameters
x0=0.5
gamma=0.25
h=0.25
e=0.15
#s=12.37
#nu=55.56*10**3
#x0exp=69.0
#gamma, h,x0 = convert2theory(s=s, nu=nu,x0exp=x0exp)
#gamma, h = convert2theory(s=s, nu=nu)

# quasi-momentum
beta0=0.0*h
Ndbeta=6.0

# spectrum
hmin=0.25
hmax=0.45
nstates=4

# explore
emin=0.42
emax=0.44
gammamin=0.25
gammamax=0.30

# free propagation/classical
iperiod=100

np.savez(inputfile,"w", description=description, N=N, e=e, gamma=gamma, h=h, x0=x0, beta0=beta0, Ndbeta=Ndbeta, iperiod=iperiod,hmin=hmin,hmax=hmax,nstates=nstates,emin=emin,emax=emax,gammamin=gammamin,gammamax=gammamax)
