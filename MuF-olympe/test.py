import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.sawtooth import *


h=2*np.pi
alpha=0.9
N=128

pot=PotentialST(alpha)
grid=Grid(N,h,xmax=2*np.pi)

#~ fo=CATFloquetOperator(grid,pot,randomphase=True)
#~ fo.diagonalize()

#~ for i in range(0,N):
	#~ print(np.abs(fo.getEvec(i).x[i])**2,N/(N*grid.ddx)**2)
				
fo=CATFloquetOperator(grid,pot,randomphase=True)
fo.diagonalize()

wf=WaveFunction(grid)
wf.setState("diracx",i0=int(N/4))

print(grid.ddp)

# ~ momenta=0
# ~ for iN in range(0,N):
	# ~ momenta+=fo.getEvec(iN).getMomentum("p",0)
# ~ momenta/=N**2

for i in range(0,N):
	fig, ax = plt.subplots(2)
	ax[0].set_xlabel(r"x")
	ax[0].set_ylabel(r"$\|\Psi(x)\|^2$")
	ax[0].plot(grid.x,np.abs(fo.getEvec(i).x)**2*(N*grid.ddx),c="blue")
	ax[0].set_ylim(0.0,3.0)
	ax[0].set_xlim(-np.pi,np.pi)
	
	ax[1].set_xlabel(r"p")
	ax[1].set_ylabel(r"$\|\Psi(p)\|^2$")
	ax[1].plot(np.fft.fftshift(grid.p),np.fft.fftshift(np.abs(fo.getEvec(i).p)**2)*(N*grid.ddp),c="red")
	ax[1].set_ylim(0.0,10.0)
	ax[1].set_xlim(np.fft.fftshift(grid.p)[0],np.fft.fftshift(grid.p)[N-1])
	
	plt.savefig("pic/"+strint(i)+".png", bbox_inches='tight',format="png")
	plt.close(fig)
#~ # ~ print(momenta)







