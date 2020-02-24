import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *
from scipy.linalg import logm, expm

mode=sys.argv[1]

Ncell=15
Npcell=32
N=Npcell*Ncell
nstates=Ncell

e=0.0
gamma=0.15*(1-0.5)
h=1/4.0

print(N*h/Ncell)

pot=PotentialMP(e,gamma)
grid=Grid(N,h,xmax=Ncell*2*np.pi)
fo=CATFloquetOperator(grid,pot)
fo.diagonalize()

wf=WaveFunction(grid)
wf.setState("coherent",x0=0.0,xratio=2.0)	

ind, overlaps=fo.getOrderedOverlapsWith(wf)

qE=np.zeros(Ncell)
beta=np.zeros(Ncell)
for i in range(0,Ncell):
	qE[i]=fo.qE[ind[i]]
	wfx=fo.eigenvec[ind[i]].x
	wfxt=np.roll(wfx,Npcell)
	beta[i]=np.mean(np.angle(wfxt/wfx))
	print(beta[i],np.abs(overlaps[i])**2)

np.savez("tempdata/states-hm4d0-e0d0",beta=beta,qEs=qE,h=h)
	
ax=plt.gca()
ax.scatter(beta,qE)
plt.show()
	











