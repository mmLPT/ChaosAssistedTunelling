import sys
sys.path.insert(0, '..')
import numpy as np
# ~ from utils.toolsbox import *

inputfile="input/e0d50-g0d15-hm2d416"

# general information
Ncell=1001
N=2*64
description=""
Npcell=32
ncellini=1

# potential parameters
e=0.50
gamma=0.15
h=1/2.416
x0=0.0
# explore
hmin=1/10
hmax=1/1


# free propagation/classical
iperiod=1000
icheck=10

np.savez(inputfile,"w", description=description, Ncell=Ncell, e=e, gamma=gamma, h=h, x0=x0, iperiod=iperiod,hmin=hmin,hmax=hmax,N=N,Npcell=Npcell,ncellini=ncellini,icheck=icheck)
