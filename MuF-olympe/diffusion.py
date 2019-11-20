import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.plot.latex import *
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
	alpha=data['alpha']
	beta=data['beta']
	N=int(data['N'])
	i0=int(data['i0'])
	tmax=int(data['tmax'])
	tcheck=int(data['tcheck'])
	data.close()
	
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params",alpha=alpha,N=N,i0=i0,tmax=tmax,nruns=nruns,tcheck=tcheck)
	

	wfx=np.zeros((int(tmax/tcheck),N))
	wfp=np.zeros((int(tmax/tcheck),N))
	p2=np.zeros(tmax)
	pro=np.zeros(tmax)
	
	pot=PotentialGG(beta,alpha)
	grid=Grid(N,h=2*np.pi,xmax=2*np.pi)

	wf=WaveFunction(grid)
	wf.setState("diracp",i0=i0)

	fo=CATFloquetOperator(grid,pot,randomphase=True)
	
	if runid==0:
		ax=plt.gca()
		ax.set_xlabel(r"x")
		ax.set_ylabel(r"V(x)")
		ax.plot(grid.x,pot.Vx(grid.x),c="blue")
		plt.savefig(wdir+"pot.png", bbox_inches='tight',format="png")
		plt.clf()
	
	for it in range(0,tmax):
		p2[it]=wf.getp2()
		pro[it]=np.abs(wf.p[i0])**2
		
		if it%tcheck==0:
			wfx[int(it/tcheck)]=np.abs(wf.x)**2
			wfp[int(it/tcheck)]=np.abs(wf.p)**2
			
		fo.propagateRandom(wf)
		
		
		
	
	np.savez(wdir+"raw-data/"+str(runid),wfx=wfx,wfp=wfp,p2=p2,pro=pro)
	
if mode=="average":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	tmax=int(data['tmax'])
	tcheck=int(data['tcheck'])
	N=int(data['N'])
	i0=int(data['i0'])
	data.close()
	
	nruns2=int(sys.argv[3])-1
	runid=int(sys.argv[4])-1 # Id of the current run
	
	istart=int(runid*int(nruns/nruns2))
	iend=istart+int(nruns/nruns2)
	
	p2=np.zeros(tmax)
	wfx=np.zeros((int(tmax/tcheck),N))
	wfp=np.zeros((int(tmax/tcheck),N))
	pro=np.zeros(tmax)
	
	for i in range(istart,iend):
		data=np.load(wdir+"raw-data/"+str(i)+".npz")
		p2+=data['p2']
		pro+=data['pro']
		wfx+=data['wfx'] 
		wfp+=data['wfp'] 
		data.close()
	
	np.savez(wdir+"partial-averaged/"+str(runid),t=np.arange(0,tmax),wfx=wfx,wfp=wfp,p2=p2,pro=pro)
	
	
if mode=="average2":
	data=np.load(wdir+"params.npz")
	nruns=int(data['nruns'])
	tmax=int(data['tmax'])
	tcheck=int(data['tcheck'])
	N=int(data['N'])
	i0=int(data['i0'])
	data.close()
	
	nruns2=179
	
	p2=np.zeros(tmax)
	pro=np.zeros(tmax)
	wfx=np.zeros((int(tmax/tcheck),N))
	wfp=np.zeros((int(tmax/tcheck),N))
	
	for i in range(0,nruns2):
		data=np.load(wdir+"partial-averaged/"+str(i)+".npz")
		p2+=data['p2'] 
		wfx+=data['wfx'] 
		pro+=data['pro']
		wfp+=data['wfp'] 
		data.close()
	
	np.savez(wdir+"averaged",p2=p2/nruns,t=np.arange(0,tmax),wfx=wfx/nruns,wfp=wfp/nruns,pro=pro/nruns)
	
if mode=="plot":
	data=np.load(wdir+"params.npz")
	alpha=data['alpha']
	N=int(data['N'])
	i0=int(data['i0'])
	tmax=int(data['tmax'])
	tcheck=int(data['tcheck'])
	nruns=int(data['nruns'])
	data.close()
	
	
	
	data=np.load(wdir+"averaged.npz")
	p2=data['p2']
	pro=data['pro']
	wfx=data['wfx']
	wfp=data['wfp']
	t=data['t']
	data.close()
	
	grid=Grid(N,h=2*np.pi,xmax=2*np.pi)
	
	for i in range(0,int(tmax/tcheck)):
		
		fig, ax = plt.subplots(2)
		ax[0].set_xlabel(r"x")
		ax[0].set_ylabel(r"$\|\Psi(x)\|^2$")
		ax[0].set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,float(alpha),nruns))
		ax[0].plot(grid.x,wfx[i,:]*(N*grid.ddx),c="blue")
		ax[0].set_ylim(0.0,5.0)
		ax[0].set_xlim(-np.pi,np.pi)
		
		ax[1].set_xlabel(r"p")
		ax[1].set_ylabel(r"$\|\Psi(p)\|^2$")
		ax[1].plot(np.fft.fftshift(grid.p),np.log(np.fft.fftshift(wfp[i,:])*(N*grid.ddp)),c="red")
		#~ ax[1].set_ylim(0.0,3.0)
		ax[1].set_xlim(np.fft.fftshift(grid.p)[0],np.fft.fftshift(grid.p)[N-1])
		
		plt.savefig(wdir+"pic/"+strint(i)+".png", bbox_inches='tight',format="png")
		plt.close(fig)
		
	#~ t[0]=t[1]
	#~ p2[0]=p2[1]
		
	ax=plt.gca()
	ax.grid()
	ax.set_xlabel(r"$t$")
	ax.set_ylabel(r"$\langle p^2 \rangle$")
	ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,float(alpha),nruns))
	ax.plot(t,p2,c="blue")

	# ~ ax.plot(np.log(t/t[0]),np.log(p2/p2[0]),c="blue")
	# ~ ax.set_xlim(0,max(t))
	# ~ ax.set_xlim(0,100)
	plt.savefig(wdir+"p2.png", bbox_inches='tight',format="png")
	plt.clf()
	
	ax=plt.gca()
	ax.grid()
	ax.set_xlabel(r"$t$")
	ax.set_ylabel(r"$\langle |\psi(i0,t)|^2\rangle$")
	ax.set_title(r"$N={:.0f} \quad \alpha={:.2f} \quad nruns={:.0f}$".format(N,float(alpha),nruns))
	ax.plot(t,pro,c="blue")

	# ~ ax.plot(np.log(t/t[0]),np.log(p2/p2[0]),c="blue")
	# ~ ax.set_xlim(0,max(t))
	# ~ ax.set_xlim(0,100)
	plt.savefig(wdir+"pro.png", bbox_inches='tight',format="png")
	plt.clf()


				
	


