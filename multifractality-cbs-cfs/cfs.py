import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.sawtooth import *
from scipy.optimize import curve_fit



# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data
 
mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"raw-data")
	os.mkdir(wdir+"partial-averaged")
	os.mkdir(wdir+"pic")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	
	nruns=int(sys.argv[4]) # number of runs for a given h
	runid=int(sys.argv[5])-1 # Id of the current run
	
	data=np.load(inputfile+".npz")
	potential=data['potential']
	alpha=data['alpha']
	beta=data['beta']
	N=int(data['N'])
	
	i0=int(data['i0'])
	atNmax=int(data['atNmax'])

	data.close()
	
		
	
	
	# ~ wfx=np.zeros((int(tmax/tcheck),N))
	# ~ wfp=np.zeros((int(tmax/tcheck),N))
	
	if potential=="RS":
		pot=PotentialST(alpha)
		grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	if potential=="GG":
		pot=PotentialGG(beta,alpha*2*np.pi)
		grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
		
	
	wf=WaveFunction(grid)
	wf.setState("diracx",i0=i0)
	fo=CATFloquetOperator(grid,pot,randomphase=True)
	
	if runid==0:
		np.savez(wdir+"params",alpha=alpha,N=N,i0=i0,atNmax=atNmax,nruns=nruns,beta=beta,potential=potential)
		ax=plt.gca()
		ax.set_xlabel(r"x")
		ax.set_ylabel(r"V(x)")
		ax.plot(grid.x,pot.Vx(grid.x),c="blue")
		plt.savefig(wdir+"pot.png", bbox_inches='tight',format="png")
		plt.clf()
		
	
	wfcfs=np.zeros(int(atNmax*N/alpha))
	for it in range(0,int(atNmax*N/alpha)):
		fo.propagateRandom(wf)
		wfcfs[it]=np.abs(wf.x[i0])**2
	
	np.savez(wdir+"raw-data/"+str(runid),wfcfs=wfcfs)
	
if mode=="average":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	atNmax=int(data['atNmax'])
	N=int(data['N'])
	alpha=data['alpha']
	data.close()
	
	nruns2=int(sys.argv[3])-1
	runid=int(sys.argv[4])-1 # Id of the current run
	
	istart=int(runid*int(nruns/nruns2))
	iend=istart+int(nruns/nruns2)
	
	wfcfs=np.zeros(int(atNmax*N/alpha))
	
	for i in range(istart,iend):
		data=np.load(wdir+"raw-data/"+str(i)+".npz")
		wfcfs+=data['wfcfs']
		data.close()
	
	np.savez(wdir+"partial-averaged/"+str(runid),wfcfs=wfcfs)
	
	
if mode=="average2":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	atNmax=int(data['atNmax'])
	N=int(data['N'])
	alpha=data['alpha']
	data.close()
	
	nruns2=179
	
	wfcfs=np.zeros(int(atNmax*N/alpha))
	time=np.arange(int(atNmax*N/alpha))*alpha/N
	
	for i in range(0,nruns2):
		data=np.load(wdir+"partial-averaged/"+str(i)+".npz")
		wfcfs+=data['wfcfs']
		data.close()
	
	np.savez(wdir+"averaged",wfcfs=wfcfs/nruns,time=time)
	
if mode=="plot":
	data=np.load(wdir+"params.npz")
	alpha=data['alpha']
	potential=data['potential']
	beta=data['beta']
	N=int(data['N'])
	i0=int(data['i0'])
	nruns=int(data['nruns'])
	data.close()
	
	# ~ potential='RS'
	
	
	data=np.load(wdir+"averaged.npz")
	wfcfs=data['wfcfs']
	time=data['time']
	data.close()
	
	grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	C=wfcfs*(N*grid.ddx)-1.0
	
	ax=plt.gca()
	ax.set_xlim(0.0,10.0)
	ax.set_ylim(0.0,1.75)
	
	
	
	time[0]=time[1]
	def Cth(x,a):
		return a*np.real(2/(np.exp(1j*x*2*np.pi)*(1+1j*x*2*np.pi/alpha*(1-alpha))-1)+1)
		

	
	popt, pcov = curve_fit(Cth, time, C)
	
	# ~ ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f} \quad \Lambda_\infty={:.3f}$".format(N,float(alpha),nruns,float(popt)))

	
	ax.plot(time,C)
	ax.plot(time,Cth(time,*popt),c="red")
	
	print(*popt)
		
	plt.savefig(wdir+"Kexp.png", bbox_inches = 'tight',format="png")
	
	
	
	
	#~ for i in range(0,int(tmax/tcheck)):
		
		#~ fig, ax = plt.subplots(2)
		#~ ax[0].set_xlabel(r"x")
		#~ ax[0].set_ylabel(r"$\|\Psi(x)\|^2$")
		#~ ax[0].set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,float(alpha),nruns))
		#~ ax[0].plot(grid.x,wfx[i,:]*(N*grid.ddx),c="blue")
		#~ ax[0].set_ylim(0.0,3.0)
		#~ ax[0].set_xlim(-np.pi,np.pi)
		
		#~ ax[1].set_xlabel(r"p")
		#~ ax[1].set_ylabel(r"$\|\Psi(p)\|^2$")
		#~ ax[1].plot(np.fft.fftshift(grid.p),np.fft.fftshift(wfp[i,:])*(N*grid.ddp),c="red")
		#~ ax[1].set_ylim(0.0,1.5)
		#~ ax[1].set_xlim(np.fft.fftshift(grid.p)[0],np.fft.fftshift(grid.p)[N-1])
		
		#~ plt.savefig(wdir+"pic/"+strint(i)+".png", bbox_inches='tight',format="png")
		#~ plt.close(fig)
	
	# ~ ax=plt.gca()	
	# ~ ax.plot(grid.x,wfx[int(tmax/tcheck)-1,:]-wfx[int(tmax/tcheck)-1,i0]+1,c="blue")
	
	
	# ~ def g(x,a,b):
		# ~ return 1-a*(((np.abs(x-grid.x[i0])))**(b)+((np.abs(x+grid.x[i0])))**(b))
		
	# ~ popt, pcov = curve_fit(g,grid.x, wfx[int(tmax/tcheck)-1,:]-wfx[int(tmax/tcheck)-1,i0]+1)
	
	# ~ print(popt)
	
	# ~ ax.plot(grid.x,g(grid.x,*popt),c="blue")
	# ~ plt.show()
	# ~ plt.clf()
	
	
	
	# ~ if potential=="RS":
		# ~ def f(x,a):
			# ~ x=x*2*np.pi/N
			# ~ return a*np.real(2/(np.exp(1j*x*alpha)*(1+1j*x*(1-alpha))-1)+1)
		
		# ~ t[0]=t[1]
	
		# ~ popt, pcov = curve_fit(f, t, contrast,bounds=(0, [3.]))
		
		# ~ ax=plt.gca()
		# ~ ax.set_xlabel(r"at/N")
		# ~ ax.grid()
		# ~ ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f} \quad \Lambda_\infty={:.3f}$".format(N,float(alpha),nruns,float(popt)))
		# ~ ax.plot(alpha*t/N,f(t,*popt),c="red",label=r"$fit\ \Lambda(t)=\Lambda_\infty\times\Re\left(1+\frac{2}{\exp\left(ia\frac{2\pi}{N}t\right)\left(1+i\frac{2\pi}{N}t(1-a)\right)-1}\right)$")
		# ~ ax.plot(alpha*t/N,(contrast),c="blue",zorder=0,label=r"$\Lambda(t)$")
		
		# ~ ax.legend()
		# ~ plt.savefig(wdir+"height-cfs.png", bbox_inches='tight',format="png")
		# ~ plt.clf()
		
	# ~ if potential=="GG":
		# ~ ax=plt.gca()
		# ~ ax.set_xlabel(r"t")
		# ~ ax.set_xlim(0.0,tmax)
		# ~ ax.set_ylim(0.0,1.0)
		# ~ ax.grid()
		# ~ ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad \beta={:.3f} \quad nruns={:.0f} $".format(N,float(alpha),float(beta),nruns))
		# ~ ax.plot(t,(contrast),c="blue",zorder=0,label=r"contraste CFS : $\frac{pic\ -\ bruit}{bruit}$")
		# ~ ax.legend()
		# ~ plt.savefig(wdir+"height-cfs.png", bbox_inches='tight',format="png")
		# ~ plt.clf()
	

				
	


