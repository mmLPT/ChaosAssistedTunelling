import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.doublewell import *
from utils.systems.general import *
from utils.toolsbox import *
import utils.latex as latex


SPSclassfile="tempdata/PP/e0d50-g0d25"


e=0.5
gamma=0.25
h=1/10.9062 #t=210
h=1/10.9065
# ~ h=1/10.828 #t=1570
x0=1.1




# ~ e=0.242
# ~ gamma=0.25
# ~ h=0.197898
# ~ x0=np.pi/2




# Simulation parameters
N=128 # number of grid points
tmax=1450
itmax=int(tmax/2) # number of period simulated (*2)
icheck=10
xm=0.0 # if you want a thrid observable centered between - and + xm

print(N*h)

# Creation of objects
pot=PotentialMP(e,gamma)
# ~ pot=PotentialDW(gamma)
grid=Grid(N,h)
husimi=Husimi(grid)
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)
xL=np.zeros(itmax)
xR=np.zeros(itmax)
xM=np.zeros(itmax)
time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
husimi=Husimi(grid,pmax=4.0,scale=5.0)

wf.setState("coherent",x0=-x0,xratio=2.0)

# ~ fo.diagonalize()
# ~ ind, overlaps=fo.getOrderedOverlapsWith(wf)


# ~ wf.x=0.5*(fo.eigenvec[ind[0]].x+fo.eigenvec[ind[1]].x)
# ~ wf.x2p()
# ~ plt.plot(grid.x,np.real(fo.eigenvec[ind[0]].x))
# ~ plt.plot(grid.x,np.real(fo.eigenvec[ind[1]].x))
# ~ plt.plot(grid.x,np.abs(wf.x)**2)
# ~ plt.show()



### SIMULATION ###

# 1 - Creation of the initial state
# ~ wf.setState("coherent",x0=-x0,xratio=2.0)
cmap = plt.cm.get_cmap('RdBu_r')

# 2 - Propagation
for it in range(0,itmax):
	print(it)
	if it%icheck==0:
		# ~ husimi.save(wf,"movie/"+strint(it/icheck))
		husimi.npz2png2(wf,"movie-cat/"+strint(it/icheck),cmapl=cmap,SPSclassbool=True,SPSclassfile=SPSclassfile,x0=grid.x)
		
	fo.propagate(wf)
	xL[it]=wf.getxM(-np.pi,-xm)
	xR[it]=wf.getxM(xm,np.pi)
	xM[it]=wf.getxM(-xm,xm)
	
	
		

# ~ # 3 - Normalization
norm=xR[0]+xL[0]+xM[0]
xL=xL/norm
xR=xR/norm
xM=xM/norm



	
plt.plot(time,xL,c="red")
plt.plot(time,xR,c="blue")
ax=plt.gca()
# ~ s,nu,x0exp=convert2exp(gamma,h,x0)
# ~ ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f} \quad h={:.3f} \quad x0={:.1f}$".format(e,gamma,h,x0)+"\n"+r"$s={:.3f} \quad \nu={:.3f} kHz \quad x_0={:.1f}^o$".format(s,nu/10**3,x0exp))
ax.set_xlim(0,tmax)
ax.set_ylim(0.0,1.0)
plt.show()


