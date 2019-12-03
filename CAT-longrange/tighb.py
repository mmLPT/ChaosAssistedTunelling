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

# ~ wdir="tempdata/states-hm4d0-e0d0.npz"
# ~ wdir="../../../data/longrange/vreg/EBETA-e0d00-g0d075-hm4d0/gathered.npz"
wdir="../../../data/longrange/vchaotic/EBETA-e0d50-g0d15-hm3d781/gathered.npz"

itmax=1000

h=1/4.0

# ~ itmax=int(itmax/2)

data=np.load(wdir)
beta=data['beta']
qEs=data['qEs']
h=data['h']
data.close()


beta=beta[:,-1]/h[:,-1]
qEs=qEs[:,-1]
h=h[0,0]
print(1/h)

print(beta.size,qEs.size)

Ncell=qEs.size

qEs=qEs[np.argsort(beta)]
V=np.fft.rfft(qEs)/(Ncell)



print()

ax=plt.gca()
ax.scatter(beta,qEs)
plt.show()

ax=plt.gca()
ax.set_yscale('log')
ax.scatter(np.arange(V.size),np.abs(V)**2)
plt.show()
	
H=np.zeros((Ncell,Ncell),dtype=complex)

def d(i,j):
	if np.abs(i-j)<int(0.5*(Ncell-1)):
		return np.abs(i-j)
	else:
		return Ncell-1-np.abs(i-j)
	
for i in range(0,Ncell):
	for j in range(i,Ncell):
		# ~ print(i,j,d(i,j))
		H[i,j]=V[d(i,j)]
		H[j,i]=np.conjugate(H[i,j])
			

U=expm(-1j*H*4*np.pi/h)
print(U)
X=np.zeros(Ncell)
n0=int(0.5*(Ncell-1))
X[n0]=1

prob=np.zeros((itmax,Ncell),dtype=complex)
xstd=np.zeros(itmax)
time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
indcell=np.arange(Ncell)


for it in range(0,itmax):
	# ~ print(it,"/",itmax)
	
	prob[it]=X
	# ~ print(np.vdot(X,X))

	xstd[it]=np.sqrt(np.sum((np.arange(Ncell)-n0)**2*np.abs(prob[it])**2)/np.sum(np.abs(prob[it])**2))
	X=np.matmul(U,X)
	# ~ print(X)



ax=plt.gca()
data=np.load("tempdata/hm3d781.npz")
time0=data['time']
indIni=data['indIni']
prob0=data['prob']
data.close()

probIni=np.sum(prob0[:,indIni],axis=1)

ax.plot(2*time0,probIni,c="blue",label="Système à N cellules")
ax.scatter(time,np.abs(prob[:,n0])**2,c="red",label="Modèle tight-binding")

ax.legend()
ax.set_xlabel("time")
ax.set_ylabel("probabilité pic initial")
plt.show()


	
# ~ ax=plt.gca()
# ~ ax.scatter(time,xstd)
# ~ plt.show()










