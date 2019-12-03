import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="input/e0d59-g0d225"

# general information
N=64
description=""

# potential parameters
e=0.59
gamma=0.225
h=0.343
x0=1.8
# ~ s=13.4
# ~ nu=48.03*10**3
# ~ x0exp=28.6
# ~ gamma, h, x0 = convert2theory(s=s, nu=nu,x0exp=x0exp)
# ~ x0=1.8

# quasi-momentum
beta0=0.0*h
Ndbeta=12.0
ibeta=10

# spectrum
nstates=4

# explore
hmin=0.2
hmax=0.4
emin=0.42
emax=0.44
gammamin=0.26
gammamax=0.32
gmin=0.0
gmax=1.0

# free propagation/classical
iperiod=1000

np.savez(inputfile,"w", description=description, N=N, e=e, gamma=gamma, h=h, x0=x0, beta0=beta0, Ndbeta=Ndbeta, iperiod=iperiod,hmin=hmin,hmax=hmax,nstates=nstates,emin=emin,emax=emax,gammamin=gammamin,gammamax=gammamax,gmin=gmin,gmax=gmax,ibeta=ibeta)
