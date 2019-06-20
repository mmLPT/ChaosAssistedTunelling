import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *

nstates=3
N=64
gamma=0.275
e=0.05/gamma
h=0.05
x0=1.05

e=0.418
gamma, h,x0 = convert2theory(s=12.60, nu=55.56*10**3,x0exp=35.0)
x0=1.2
print(x0)

pot=PotentialMP(e,gamma)
grid=Grid(N,h)
husimi=Husimi(grid,pmax=4.0)
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)

# 1 : on diagonalise Floquet
fo.diagonalize()

wf=WaveFunction(grid)
wf.setState("coherent",x0=x0,xratio=2.0)
fo.computeOverlapsAndQEs(wf)
qes, overlaps, symX, ind=fo.getQEsOverlapsSymmetry(nstates,True)


nu=np.zeros((nstates,nstates))
evec=[]

for i in range(0,nstates):
	evec.append(fo.getEvec(ind[i],False))
	husimi.save(evec[i],"data/evec"+strint(i))
	for j in range(0,nstates):
		nu[i,j]=np.abs(fo.diffqE1qE2(ind[i],ind[j]))/(h)

np.savez("states",symX=symX,overlaps=overlaps,qes=qes,x=grid.x,nu=nu,evec=evec)
# 5- pour chacun des n états







