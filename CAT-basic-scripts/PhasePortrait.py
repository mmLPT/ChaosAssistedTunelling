import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *


gamma=0.275
e=0.05/gamma
e=0.417
gamma, h,x0 = convert2theory(s=12.60, nu=55.56*10**3,x0exp=35.0)
pot=PotentialMP(e,gamma)
nperiod=300
ny0=100
wdir="data/"

xmax=np.pi
pmax=2.0
cp=ClassicalContinueTimePropagator(pot)
sb=PhasePortrait(nperiod,ny0,cp,xmax=xmax,pmax=pmax) 

sb.save(wdir)
sb.npz2png(wdir)
