import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.kickedrotor import *
from utils.systems.general import *
from utils.toolsbox import *
import utils.latex as latex
from scipy.special import gamma




N=1024# number of grid points
h=2*np.pi/N
smax=4

# Creation of objects
pot=PotentialKR(7.0)
grid=Grid(N,h)
fo=CATFloquetOperator(grid,pot)
husimi=Husimi(grid,pmax=2*np.pi,scale=5.0)

fo.diagonalize()
ps, bins=fo.getSpacingDistribution2(bins=25)
np.savez("data/spacings",ps=ps,bins=bins)

ps=np.load("data/spacings"+".npz")["ps"]
bins=np.load("data/spacings"+".npz")["bins"]
s=np.linspace(0,smax,10000)
psgoe=0.5*np.pi*s*np.exp(-0.25*np.pi*s**2)
pspoisson=np.exp(-s)

ax=plt.gca()
ax.set_xlabel(r"s")
ax.set_ylabel(r"P(s)")
ax.set_xlim(0,smax)
ax.set_ylim(0,1)
ax.plot(s,psgoe,color="red",label="GOE",lw=3.0,ls="--")
ax.plot(s,pspoisson,color="blue",label="Poisson",lw=3.0,ls="--")
ax.bar((bins[:-1] + bins[1:]) / 2, ps, align='center', width=1.0 * (bins[1] - bins[0]),edgecolor="black",color="silver")
ax.legend()
plt.show()

# ~ for i in range(0,N):
	# ~ print(i+1,"/",N)
	# ~ husimi.save(fo.getEvec(i),"data/"+strint(i))
	# ~ husimi.npz2png("data/"+strint(i),cmapl="Reds",SPSclassbool=True,SPSclassfile="data/all-trajectories.npz")



