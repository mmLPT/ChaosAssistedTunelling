import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *


Ncell=1
N=64*Ncell
nstates=5

# Run magique
# ~ e=0.50
# ~ gamma=0.25
# ~ h=1/10.9062
# ~ h=1/10.855
# ~ x0=1.1

# ~ gamma=0.375
# ~ e=0.245
# ~ x0=1.8
# ~ h=1/3.1

# ~ gamma=0.315
# ~ e=0.400
# ~ x0=1.8
# ~ h=1/4.3

gamma=0.225
e=0.59
x0=1.6
h=0.3


SPSclassfile="tempdata/PP/e0d24-g0d375"
SPSclassfile="tempdata/PP/e0d40-g0d315"
SPSclassfile="tempdata/PP/e0d59-g0d225"
wdir="tempdata/husimi/"
cmap = plt.cm.get_cmap('Purples')


pot=PotentialMP(e,gamma)
grid=Grid(N,h,xmax=Ncell*2*np.pi)
husimi=Husimi(grid,pmax=N*h,scale=5.0)
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)
wf.setState("coherent",x0=x0,xratio=2.8)

# 1 : on diagonalise Floquet
fo.diagonalize()
ind, overlaps=fo.getOrderedOverlapsWith(wf)


husimi.save(wf,wdir+"initial")
husimi.npz2png(wdir+"initial",cmapl=cmap,SPSclassbool=True,SPSclassfile=SPSclassfile,textstr="Initial")		

for i in range(0,nstates):
	print(i,"/",nstates,np.abs(overlaps[i])**2)
	husimi.save(fo.getEvec(ind[i]),"tempdata/husimi/"+strint(i))
	qE=4*np.pi*fo.getQE(ind[i])/(h*2*np.pi)
	# ~ if qE<0:
		# ~ qE=qE+1
	if fo.getEvec(ind[i]).isSymetricInX():
		cmap = plt.cm.get_cmap('Blues')
	else:
		cmap = plt.cm.get_cmap('Reds')
		
	if np.abs(overlaps[i])**2<0.25:
		cmap = plt.cm.get_cmap('Greens')
		
	husimi.npz2png("tempdata/husimi/"+strint(i),cmapl=cmap,SPSclassbool=True,SPSclassfile=SPSclassfile,textstr="{:.1f}".format(100*np.abs(overlaps[i])**2)+r"$\% \quad qE={:.3f}\pi$".format(qE))
		









