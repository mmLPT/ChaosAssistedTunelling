import sys
sys.path.insert(0, '..')
import numpy as np
from utils.toolsbox import *

inputfile="data/input/s-27.84-nu-80.0-Ndbeta-8.0-beta0=0.5"

# general information
N=64
description="free propagation averaged"

# potential parameters
e=0.44
x0=35.0*np.pi/180.0
#gamma=0.305
#h=0.250
s=27.84
nu=80.0*10**3
gamma, h = modesbasic.convert2theory(s=s, nu=nu)

# quasi-momentum
beta0=0.5*h
Ndbeta=8.0
dbeta=h/(3.0*Ndbeta)

# i/o
iperiod=3000
icheck=1

np.savez(inputfile,"w", description=description, N=N, e=e, gamma=gamma, h=h, x0=x0, beta0=beta0, dbeta=dbeta, Ndbeta=Ndbeta, iperiod=iperiod, icheck=icheck)
