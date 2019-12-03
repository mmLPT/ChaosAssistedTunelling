import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *
import matplotlib.gridspec as gridspec


e=0.5
gamma=0.25
h=1/10.9062
h=1/10.855
x0=1.4

# ~ e=0.24
# ~ gamma=0.370
# ~ h=1/2.918
# ~ x0=1.0



gamma=0.231
e=0.6014
x0=24.6/np.pi*2
h=1/3.4


nstates=7
N=64
print(N*h,2*np.pi)
itmax=65
xm=0.2


freq=np.zeros((nstates,nstates))
evec=[]
xL=np.zeros(itmax)
xR=np.zeros(itmax)
xM=np.zeros(itmax)


pot=PotentialMP(e,gamma)
grid=Grid(N,h)
husimi=Husimi(grid,pmax=4.0,scale=5.0,cmapl='Blues',SPSclassfile='data/all-trajectories.npz')
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)
wf.setState("coherent",x0=x0,xratio=2.0)

husimi.save(wf,"wf0")


# 1 : on diagonalise Floquet
fo.diagonalize()
ind, overlaps=fo.getOrderedOverlapsWith(wf)

for i in range(0,nstates):
	evec.append(fo.getEvec(ind[i]))
	#husimi.save(evec[i],"data/evec"+strint(i))
	for j in range(0,nstates):
		freq[i,j]=fo.getTunnelingFrequencyBetween(ind[i],ind[j])
		print(i,j,freq[i,j])
		
evecx=np.zeros((nstates,N))
for i in range(0,nstates):
	evecx[i]=np.real(evec[i].x)
	husimi.save(evec[i],"data/"+str(i))
	
plt.clf()

wfd=WaveFunction(grid)
wfd=overlaps[0]*evec[0]+overlaps[1]*evec[1]++overlaps[2]*evec[2]

# ~ plt.plot(grid.x,np.abs(wfd.x)**2)
# ~ plt.show()

# 2 - Propagation
time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
# ~ for it in range(0,itmax):
	# ~ fo.propagate(wf)
	# ~ xL[it]=wf.getxM(-np.pi,-xm)
	# ~ xR[it]=wf.getxM(xm,np.pi)
	# ~ xM[it]=wf.getxM(-xm,xm)		

# 3 - Normalization
norm=xR[0]+xL[0]+xM[0]
xL=xL/norm
xR=xR/norm
xM=xM/norm


freqfft=np.fft.rfftfreq(time.size,d=2.0)
xLfft=np.abs(np.fft.rfft(xL))
xRfft=np.abs(np.fft.rfft(xR))
xLfft[0]=0.0
xRfft[0]=0.0

# PLOT
lw=2.0
gs1 = gridspec.GridSpec(1,3)

ax =plt.subplot(gs1[0])	
# ~ ax=plt.gca()
ax.set_xlim(0.0,max(time))
ax.set_ylim(0.0,1.0)

ax.plot(time,xL,c="red")
ax.plot(time,xR,c="blue")	
ax.plot(time,xM,c="yellow")

ax =plt.subplot(gs1[1])
ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f} \quad h={:.3f} \quad x0={:.1f}$".format(e,gamma,h,x0)+"\n"+r"$s={:.3f} \quad \nu={:.3f} kHz \quad x_0={:.1f}^o$".format(s,nu/10**3,x0exp))
ax.set_xlim(0.0,max(freqfft))
ax.set_ylim(0.0,1.0)	
ax.plot(freqfft,0.5*xRfft/max(xRfft),lw=lw,c="black")
ax.legend()
for i in range(0,nstates):
	for j in range(i,nstates):
		if j != i:
			ax.plot((freq[i,j],freq[i,j]),(0.0,1.0),label=str(i)+"//"+str(j),lw=lw,ls="--")
			
ax=plt.subplot(gs1[2])	
for i in range(0,nstates):		
	ax.plot(grid.x,np.real(evec[i].x),label=str(i)+r"$\quad |\langle {:.0f}|\psi \rangle|^2=${:.2f}".format(i,np.abs(overlaps[i])**2))
	ax.set_xlim(-np.pi,np.pi)
	ax.set_ylim(-1.0,1.0)
ax.legend()
			
# SAVE
	
np.savez("states",overlaps=overlaps,x=grid.x,freq=freq,evecx=evecx,nstates=nstates,xL=xL,xR=xR,xM=xM,time=time)

# ~ plt.show()




