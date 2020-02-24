import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="input/g0d375-e0d24"

# general information
N=64
Ncell=151
Nini=13
description=""

# potential parameters
e=0.24
gamma=0.375
h=0.343
x0=np.pi/2
# ~ s=13.4
# ~ nu=48.03*10**3
# ~ x0exp=28.6
# ~ gamma, h, x0 = convert2theory(s=s, nu=nu,x0exp=x0exp)
# ~ x0=1.8

# quasi-momentum
beta0=0.0*h
Ndbeta=12.0
ibeta=10

# explore
hmin=1/3.4
hmax=1/2.4

# free propagation/classical
iperiod=500

np.savez(inputfile,"w", 
	description=description, 
	N=N, 
	Ncell=Ncell,
	Nini=Nini,
	e=e, 
	gamma=gamma, 
	h=h, 
	x0=x0, 
	beta0=beta0, 
	Ndbeta=Ndbeta, 
	iperiod=iperiod,
	hmin=hmin,
	hmax=hmax)
