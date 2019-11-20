import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.systems.kickedrotor import *


# ~ gamma=0.375
# ~ e=0.242
# ~ h=1/2.88
# ~ x0=1.8

# ~ e=0.400
# ~ gamma=0.310
# ~ h=1/4.53
# ~ x0=1.8/4.5*np.pi

e=0.00
gamma=0.15
h=1/1.66
x0=0.0

xratio=2.5

# ~ K=1.5
tmax=25


Ncell=51
ncellini=1
ncheck=5

km=int(Ncell/2)
# ~ km=15


# Simulation parameters
N=Ncell*32# number of grid points
itmax=int(tmax/2) # number of period simulated (*2)
icheck=2

# ~ h=2*np.pi/N
print(N*h/Ncell)


SPSclassfile="tempdata/PP"

# Creation of objects
pot=PotentialMP(e,gamma)
grid=Grid(N,h,xmax=Ncell*2*np.pi)
husimi=Husimi(grid)
fo=CATFloquetOperator(grid,pot)

time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
husimi=Husimi(grid)

### SIMULATION ###

# 1 - Creation of the initial state

wf=WaveFunction(grid)
wf.setState("coherent",x0=x0,xratio=xratio)
	
ncell2=int((ncellini-1)*0.5)
for i in range(1,ncell2+1):
	xi=x0+i*2*np.pi
	wfi=WaveFunction(grid)
	wfi.setState("coherent",x0=xi,xratio=xratio)
	wf=wf+wfi
	xi=x0-i*2*np.pi
	wfi=WaveFunction(grid)
	wfi.setState("coherent",x0=xi,xratio=xratio)
	wf=wf+wfi

wf.normalizeX()
wf.x2p()


fo.propagate(wf)

		
Ueff=np.zeros((Ncell,Ncell),dtype=complex)	
V=np.zeros(Ncell,dtype=complex)	
for i in range(0,Ncell):
	x=x0+(i-int(Ncell/2))*2*np.pi
	wfi=WaveFunction(grid)
	wfi.setState("coherent",x0=x,xratio=xratio)
	V[i]=wf%wfi
	
for i in range(0,Ncell):	
	for j in range(i,Ncell):
		l=int(min(np.abs(i-j),np.abs(Ncell+i-j)))
		Ueff[i,j]=V[l+int(Ncell/2)]
		Ueff[j,i]=np.conjugate(Ueff[i,j])

wf=WaveFunction(grid)
wf.setState("coherent",x0=x0,xratio=xratio)


		
Peff=np.zeros((tmax,int(Ncell/2+1)),dtype=complex)	


		
P=np.zeros((tmax,int(Ncell/2+1)),dtype=complex)		
for i in range(0,tmax):
	print(i)
	fo.propagate(wf)
	Ueffy=np.linalg.matrix_power(Ueff,i+1)
	for j in range(0,int(Ncell/2+1)):
		x=x0+j*2*np.pi
		wfi=WaveFunction(grid)
		wfi.setState("coherent",x0=x,xratio=xratio)
		P[i,j]=wf%wfi
		Peff[i,j]=Ueffy[0,j]
	


	
# ~ ax=plt.subplot(2,1,1)
ax=plt.gca()
ax.set_yscale('log')
ax.set_ylim(10**(-8),1)
ax.set_xlim(0,25)
tc=5
for t in range(0,int(tmax/tc)):
	print(tc*t)
	ax.plot(np.arange(P[tc*t].size),np.abs(P[tc*t])**2)
ax.grid()

# ~ ax=plt.subplot(2,1,2)
# ~ ax.set_yscale('log')
# ~ ax.set_ylim(10**(-8),1)
# ~ ax.set_xlim(0,25)
# ~ tc=5
# ~ for t in range(0,int(tmax/tc)):
	# ~ ax.plot(np.arange(Peff[tc*t].size),np.abs(Peff[tc*t])**2)
# ~ ax.grid()

plt.show()



# 2 - Propagation

# ~ xm=np.zeros(itmax)
# ~ a=np.zeros(itmax)
# ~ b=np.zeros((ncheck,itmax))
# ~ p=np.zeros(itmax)
# ~ for it in range(0,itmax):
	# ~ if it%icheck==0:
		# ~ print(it)
		# ~ husimi.save(wf,"tempdata/"+strint(int(it/icheck)*2))
		# ~ husimi.npz2png("tempdata/"+strint(int(it/icheck)*2),cmapl="plasma",SPSclassbool=True,SPSclassfile=SPSclassfile)
		
		# ~ ax=plt.gca()
		# ~ ax.plot(grid.x,np.abs(wf.x)**2)
		# ~ ax.set_xlim(-xmax/2,xmax/2)
		# ~ ax.set_ylim(0.0,0.25)
		# ~ ax.set_xticks(np.linspace(-xmax/2,xmax/2,np.ceil(0.5*xmax/np.pi)+1,endpoint=True),minor=True)
		# ~ ax.set_xticklabels([])
		# ~ ax.grid(which='minor', color="red",alpha=1.0)	
		# ~ plt.savefig("tempdata/wf3/N40-"+strint(it))
		# ~ ax.clear()
		
	# ~ fo.propagatequater(wf)
	# ~ p[it]=wf.getp()
	# ~ fo.propagatequater2(wf)
	# ~ xm[it]=wf.getxstd()
	# ~ for ic in range(0,ncheck):
		# ~ x0=ic*3*2*np.pi
		# ~ b[ic,it]=wf.getxM(x0-np.pi,x0)
	# ~ a[it]=wf.getxM(-np.pi+(Ncell-1)*np.pi,np.pi+(Ncell-1)*np.pi)
	
		
		
		
# ~ np.savez("tempdata/2-Ncell-151-hm-4d53-N40",xm=xm,time=time,a=a,x=grid.x,b=b,p=p)		

	
	




