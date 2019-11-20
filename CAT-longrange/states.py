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

wdir="tempdata/states/"
SPSclassfile=wdir+"PP"


Ncell=15
N=32*Ncell
nstates=Ncell

e=0.5
gamma=0.15
h=0.3

print(N*h/Ncell)


pot=PotentialMP(e,gamma)
grid=Grid(N,h,xmax=Ncell*2*np.pi)
fo=CATFloquetOperator(grid,pot)
# ~ fo.diagonalize()

# ~ wfcs=WaveFunction(grid)
# ~ wfcs.setState("coherent",x0=0.0,xratio=2.0)
# ~ ind, overlaps=fo.getOrderedOverlapsWith(wfcs)


# ~ for i in range(0,nstates):
	# ~ print(i,"/",nstates,np.abs(overlaps[i])**2)
	# ~ evec.append(fo.getEvec(ind[i]))
	# ~ husimi.save(evec[i],wdir+strint(i))
	# ~ qE=4*fo.getQE(ind[i])/h
	# ~ if fo.getEvec(ind[i]).isSymetricInX():
		# ~ husimi.npz2png(wdir+strint(i),cmapl="Blues",SPSclassbool=True,SPSclassfile=SPSclassfile,textstr="{:.1f}".format(100*np.abs(overlaps[i])**2)+r"$\% \quad qE={:.3f}\pi$".format(qE))
	# ~ else:
		# ~ husimi.npz2png(wdir+strint(i),cmapl="Reds",SPSclassbool=True,SPSclassfile=SPSclassfile,textstr="{:.1f}".format(100*np.abs(overlaps[i])**2)+r"$\% \quad qE={:.3f}\pi$".format(qE))
		

# ~ for j in range(0,Ncell):
	# ~ for i in range(0,nstates):
		# ~ x0=-3*np.pi+j*2*np.pi
		# ~ print(x0/np.pi)
		# ~ wf=WaveFunction(grid)
		# ~ wf.setState("coherent",x0=x0,xratio=2.0)
		# ~ print(j,i,fo.getEvec(ind[i])%wf)
	
			
# ~ ax=plt.gca()
# ~ for i in range(0,nstates):
	# ~ print(i,"/",nstates,overlaps[i])
	# ~ wfi=fo.getEvec(ind[i])
	# ~ ax.plot(grid.x,np.real(wfi.x))
	# ~ ax.plot(grid.x,np.imag(wfi.x))
	# ~ ax.set_title("qE="+str(fo.getQE(ind[i]))+"/ovlaps="+str(overlaps[i]))
	# ~ plt.savefig(wdir+strint(i))
	# ~ ax.clear()
	
# ~ H=np.identity(N)
# ~ for i in range(0,N):
	# ~ H[i,i]=fo.getQE(i)
	
# ~ print(H)

V=np.zeros(int((Ncell+1)*0.5),dtype=complex)
# ~ fo.fillM()

wf0=WaveFunction(grid)
wf0.setState("coherent",x0=0.0,xratio=2.0)	

# ~ wf0_nb=np.zeros(N,dtype=complex)
# ~ eigenval=np.zeros(N,dtype=complex)
# ~ for i in range(0,N):
	# ~ wf0_nb[i]=sum(np.conj(fo.getEvec(i).x)*wf0.x)*grid.ddx
	# ~ eigenval[i]=fo.getEval(i)
	

# ~ for i in range(0,N):
	# ~ print(fo.getEval(i),np.abs(fo.getEval(i))**2)

fo.fillM()
# ~ M=-1j*logm(fo.M)*h/(4*np.pi)
M=fo.M

for j in range(0,int((Ncell+1)*0.5)):
	x0=j*2*np.pi
	# ~ print(x0/np.pi)
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)
	# ~ wf_nb=np.zeros(N,dtype=complex)
	# ~ for i in range(0,N):
		# ~ wf_nb[i]=sum(np.conj(fo.getEvec(i).x)*wf.x)*grid.ddx
	# ~ V[j]=np.vdot(wf0_nb,wf_nb*eigenval)
	V[j]=np.vdot(wf0.x,M@wf.x)*grid.ddx
	print(np.abs(V[j]))
	
ax=plt.gca()
ax.set_yscale('log')
ax.scatter(np.arange(int((Ncell+1)*0.5)),np.abs(V)**2)
plt.show()
	











