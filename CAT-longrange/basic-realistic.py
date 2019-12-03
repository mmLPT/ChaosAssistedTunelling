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
# ~ e=0.24
# ~ h=0.346
# ~ x0=1.75
# ~ h=1/2.957
# ~ h=1/2.591
# ~ tmax=10

e=0.6
gamma=0.15
h=0.5
x0=0
tmax=50


Ncell=51
ncellini=13
ncheck=5

km=int(Ncell/2)

# Simulation parameters
N=Ncell*32# number of grid points
itmax=int(tmax/2) # number of period simulated (*2)
icheck=5
xratio=2.0

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


pmean=np.zeros(itmax)
pL=np.zeros(itmax)	
pR=np.zeros(itmax)	
probL=np.zeros((itmax,Ncell))	
probR=np.zeros((itmax,Ncell))
# ~ prob=np.zeros((itmax,Ncell))

xmax=30*2*np.pi
	
for i in range(0,itmax):
	
	
	
	for j in range(0,Ncell):
		xL=x0+(j-0.5*(Ncell-1))*2*np.pi
		wfL=WaveFunction(grid)
		wfL.setState("coherent",x0=xL,xratio=xratio)
		probL[i,j]=np.abs(wf%wfL)**2
		xR=-x0+(j-0.5*(Ncell-1))*2*np.pi
		wfR=WaveFunction(grid)
		wfR.setState("coherent",x0=xR,xratio=xratio)
		probR[i,j]=np.abs(wf%wfR)**2
	print(np.sum(probR[i,:]))
		
		
	# ~ if i%icheck==0:
		# ~ ax=plt.gca()
		# ~ ax.plot(grid.x,np.abs(wf.x)**2)
		# ~ ax.set_xlim(-xmax/2,xmax/2)
		# ~ ax.set_ylim(0.0,0.25)
		# ~ ax.set_xticks(np.linspace(-xmax/2,xmax/2,np.ceil(0.5*xmax/np.pi)+1,endpoint=True),minor=True)
		# ~ ax.set_xticklabels([])
		# ~ ax.grid(which='minor', color="red",alpha=1.0)	
		# ~ plt.savefig("tempdata/exp/"+strint(i))
		# ~ ax.clear()
	
	print(i,"/",itmax)
	fo.propagatequater(wf)
	pmean[i]=wf.getp()
	pL[i]=wf.getpL()
	pR[i]=wf.getpR()
	fo.propagatequater2(wf)
	
# ~ fo.propagatequater(wf)
	
# ~ ax=plt.gca()
# ~ ax.plot(np.fft.fftshift(grid.p),np.fft.fftshift(np.abs(wf.p)**2))
# ~ plt.show()
		
		
ind=np.arange(Ncell)
N0=(Ncell-1)*0.5
n0=(ncellini-1)*0.5
n1=5
indIni=np.logical_and(ind>(N0-n0-1),ind<(N0+n0+1))
# ~ indElse=(ind<(N0-n0))*(ind>(N0-n0-n1))+(ind>(N0+n0))*(ind<(N0+n0+n1))
indElse=np.logical_not(indIni)
# ~ print(ind[indIni])
# ~ print(ind[indElse])

# ~ probL[:]=/np.sum(+probR[:])

print(probL[:,indIni])

probIni=np.zeros(itmax)	
probElse=np.zeros(itmax)
for i in range(0,itmax):
	probIni[i]=np.sum(probL[i,indIni]+probR[i,indIni])
	probElse[i]=np.sum(probL[i,indElse]+probR[i,indElse])
	
		

	
	
# ~ ax=plt.subplot(3,1,1)

# ~ ax.set_ylim(0.0,1.1)
# ~ ax.set_xlim(0,np.max(time))

# ~ ini=indIni*np.arange(Ncell)
# ~ for j in ini[ini.astype(bool)] :
	# ~ print(j)
	# ~ ax.plot(time,np.abs(probL[:,j])**2,label=str(j))
	# ~ ax.plot(time,np.abs(probR[:,j])**2,label=str(j))
# ~ ax.grid()
# ~ ax.legend()

ax=plt.subplot(3,1,2)
ax.set_xlim(0,np.max(time))
ax.plot(time,probIni,c="blue")
ax.plot(time,probElse,c="red")
ax.plot(time,probIni+probElse,c="green")
ax.grid()

# ~ ax=plt.subplot(3,1,3)
# ~ ax.set_xlim(0,np.max(time))
# ~ ax.plot(time,pL,c="blue")
# ~ ax.plot(time,pR,c="red")
# ~ ax.grid()

np.savez("tempdata/46849-qm",xL=pL,xR=pR,time=time)


plt.show()



	
	




