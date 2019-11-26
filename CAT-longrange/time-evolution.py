import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.toolsbox import *
from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *

# State: stable [22/02/2019]

# To be used with "run-spectrum.slurm"

# This scripts makes possibles to 
# 1. compute in // the spectrum of qE for differents value of [h]
# 2. gather the information
# 3. plot the spectrum

# Arguments to provide:
# 1. mode = "compute", "gather", "plot"
# 2. working directory
# if mode=="compute":
# 	3. input file
# 	4. total number of tasks
# 	5. id of the current runs

mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"runs")
	os.mkdir(wdir+"velocity")
	os.mkdir(wdir+"populations")

if mode=="compute":
	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	description=data['description']  

	Ncell=int(data['Ncell'])
	Npcell=int(data['Npcell'])
	ncellini=int(data['ncellini'])
	e=data['e']
	gamma=data['gamma']
	x0=data['x0'] 
	iperiod=int(data['iperiod'])
	icheck=int(data['icheck'])
	ncheck=int(iperiod/icheck)

	hmin=data['hmin']
	hmax=data['hmax']

	data.close()
	
	N=Ncell*Npcell

	# Initialization of potential and correcting the x0 value if needed
	pot=PotentialMP(e,gamma)
	
	nruns=int(sys.argv[4]) # Total number of // runs
	runid=int(sys.argv[5])-1 # Id of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, nruns=nruns,Ncell=Ncell,N=N, e=e,gamma=gamma,x0=x0,iperiod=iperiod,hmin=hmin,hmax=hmax,ncellini=ncellini,Npcell=Npcell)


	
	ind=np.arange(Ncell)
	N0=(Ncell-1)*0.5
	n0=(ncellini-1)*0.5
	indIni=np.logical_and(ind>(N0-n0-1),ind<(N0+n0+1))
	indElse=np.logical_not(indIni)
	x0t=np.linspace(x0-0.5*(Ncell-1)*2*np.pi,x0+0.5*(Ncell-1)*2*np.pi,Ncell,endpoint=True)
	xratio=2.0

	#
	h=1/np.linspace(1.0/hmax,1.0/hmin,nruns)[runid]
	grid=Grid(N,h,xmax=Ncell*2*np.pi)
	fo=CATFloquetOperator(grid,pot)
	wf=WaveFunction(grid)
		
	for icell in ind[indIni]:
		xicell=x0t[icell]
		wficell=WaveFunction(grid)
		wficell.setState("coherent",x0=xicell ,xratio=xratio)
		wf=wf+wficell
		
	wf.normalizeX()
	wf.x2p()

	prob=np.zeros((ncheck,Ncell))
	xstd=np.zeros(ncheck)
	time=np.zeros(ncheck)
		
	for it in range(0,iperiod):
		if it%icheck==0:
			i0=int(it/icheck)
			time[int(it/icheck)]=it
			for icell in ind:
				xicell=x0t[icell]
				wficell=WaveFunction(grid)
				wficell.setState("coherent",x0=xicell,xratio=xratio)
				prob[int(it/icheck),icell]=np.abs(wf%wficell)**2

				xstd[int(it/icheck)]=wf.getxstd()
		fo.propagate(wf)
		
	probIni=np.sum(prob[:,indIni],axis=1)
	probElse=np.sum(prob[:,indElse],axis=1)
		
	ax=plt.gca()	
	ax.set_xlim(0,np.max(time))
	ax.plot(time,probIni,c="blue")
	ax.plot(time,probElse,c="red")
	ax.plot(time,probIni+probElse,c="green")
	ax.grid()
	
	plt.savefig(wdir+"populations/"+strint(runid))	

	# Save data
	np.savez(wdir+"runs/"+str(runid),"w", h=h, x=grid.x,wff=wf.x,prob=prob,time=time,indIni=indIni,indElse=indElse,xstd=xstd)

if mode=="gather":
	# This mode collect the spectrum for each value of h and make a single file

	# Reading inputfile
	data=np.load(wdir+"params.npz")
	nruns=data['nruns']
	data.close()

	# Create array to store data
	ht=np.zeros(nruns)
	vt=np.zeros(nruns)
	
	# For each runs read the output and add it to the new array
	for i in range(0,nruns):
		data=np.load(wdir+"runs/"+str(i)+".npz")
		time=data['time']
		xstd=data['xstd']
		prob=data['prob']
		h=data['h']
		data.close()
		
		fit = np.polyfit(time[0:50],xstd[0:50]/(2*np.pi), 1)
		
		vt[i]=fit[0]
		ht[i]=h
		
		fig=plt.figure()
		ax=plt.subplot(2,1,1)
		ax.scatter(time,xstd/(2*np.pi))
		ax.plot(time,fit[0]*time+fit[1],c="red")
		
		ax=plt.subplot(2,1,2)
		ax.plot(time,np.abs(prob[:,-1])**2)
		plt.savefig(wdir+"velocity/"+strint(i))
		plt.close(fig)
	# Save the array
	np.savez(wdir+"gathered","w", h=ht,v=vt)

if mode=="plot":
	# Reading inputfile
	data=np.load(wdir+"gathered.npz")
	h=data['h']
	v=data['v']
	data.close()

	# General setup for plotting
	ax=plt.gca()
	ax.set_yscale('log')
	ax.set_xlim(np.min(1/h),np.max(1/h))
	
	# ~ w=25
	# ~ vtyp=np.convolve(v, np.ones(w)/w, 'valid')
	# ~ hm=1/h
	# ~ plt.plot(hm[int(w/2-1):int(h.size-w/2)],vtyp,c="red")
	
	# ~ fit = np.polyfit(1/h[0:int(h.size*0.5)],np.log(v[0:int(h.size*0.5)]), 1)
	
	a=-1.03
	b=-0.43
	# ~ v=v-np.exp(a*1/h+b)
	
	
	plt.scatter(1/h,v)
	fit = np.polyfit(1/h,np.log(v), 1)
	ax.plot(1/h,np.exp(fit[0]*1/h+fit[1]),c="red")
	
	print(fit[0],fit[1])
	
	ax.set_ylim(10**(-3),1)


	plt.savefig(wdir+"final")
	

