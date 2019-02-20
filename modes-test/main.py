import sys
import os
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt
import modesbasic
import modesxconfinment
import modesquasimomentum
#~ import modesinitialstate

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *


if(len(sys.argv)>2):
	if(sys.argv[1]=="compute"):
		compute=True
	if(sys.argv[1]=="Ncompute"):
		compute=False
	if(sys.argv[2]=="read"):
		read2=True
	if(sys.argv[2]=="Nread"):
		read2=False
else:
	compute=False
	read2=False
	print("missing arguments")
	
wdir=sys.argv[3]
if not os.path.exists(wdir):
    os.mkdir(wdir)

N=64

########################################################################
gamma, h = modesbasic.convert2theory(s=25.84,nu=72.0*10**3)
e=0.44
x0=90.0*np.pi/180.0

#~ e=0.02
#~ gamma=0.25
#~ h=0.2

pot=PotentialMP(e,gamma)
grid=Grid(N,h)
#~ x0=pot.x0

#~ x0=35.0/180.0*np.pi


#~ modesbasic.classical(pot,nperiod=250,ny0=50,wdir="regular-tunneling/",compute=compute)
#~ modesbasic.propagate( grid, pot, compute=compute, read=read, iperiod=1000, icheck=1,wdir="quasimomentum/",datafile="blocked")
#~ modesbasic.propagate( grid, pot, compute=compute, read=read, iperiod=100, icheck=1,wdir="test/",datafile="new")

#~ modesbasic.period_with_h(e=0.29, gamma=0.29, imax=50, N=128, datafile="split")
#~ modesbasic.explore_epsilon_gamma()
#~ modesbasic.check_projection(grid,pot,iperiod)
#~ modesbasic.period_with_gamma(e, h,compute=compute,read=read2)

#~ modesxconfinment.perturbation_theory(grid,e,gamma,compute=compute=,read=read)
#~ modesxconfinment.track_crossing(N=64, e=0.315, gamma=0.290, hmin=0.298, hmax=0.302,datafile="data/track_crossing4")
#~ modesxconfinment.check_T_with_confinment(imax=220,e=e,gamma=gamma)
#~ modesxconfinment.symetry_of_gs_with_h(N=64, e=0.315, gamma=0.290, datafile="data/croisement3",compute=compute,read=read)

#~ modesquasimomentum.free_prop_averaged(grid,pot,x0,iperiod=50,ibetamax=125,wdir=wdir)
#~ modesquasimomentum.free_prop_averaged(grid,pot,x0,compute=compute,read=read,iperiod=1000,ibetamax=100,wdir=wdir) 
modesquasimomentum.free_prop_averaged(grid,pot,x0,Ndbeta=2.0,compute=compute,read=read,iperiod=100,ibetamax=50,wdir=wdir,beta0=0.0) 

#~ modesquasimomentum.distribution_omega(grid,pot,compute=compute,read=read,ibetamax=500,wdir="quasimomentum/",datafile="distribution4")
#~ modesquasimomentum.distribution_omega(grid,pot,compute=compute,read=read,ibetamax=2500,wdir="quasimomentum/",datafile="scan",scan=True)

#~ modesinitialstate.imaginary(gamma,e,h)
#~ modesinitialstate.loading(gamma,e,h)







