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
	os.mkdir(wdir+"pics")
	os.mkdir(wdir+"wf")

if mode=="compute":
	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	description=data['description']  

	Ncell=int(data['Ncell'])
	e=data['e']
	gamma=data['gamma']
	x0=data['x0'] 
	iperiod=int(data['iperiod'])

	hmin=data['hmin']
	hmax=data['hmax']

	data.close()
	
	N=Ncell*128
	time=np.linspace(0.0,2.0*iperiod,num=iperiod,endpoint=False)
	

	# Initialization of potential and correcting the x0 value if needed
	pot=PotentialMP(e,gamma)
	
	nruns=int(sys.argv[4]) # Total number of // runs
	runid=int(sys.argv[5])-1 # Id of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, nruns=nruns,Ncell=Ncell,N=N, e=e,gamma=gamma,x0=x0,iperiod=iperiod,hmin=hmin,hmax=hmax)

	# Initialization of the grid for given h value
	h=1/np.linspace(1.0/hmax,1.0/hmin,nruns)[runid]
	grid=Grid(N,h,xmax=Ncell*2*np.pi)

	# Creating array to store data
	xstd=np.zeros(iperiod)
	popNthcell=np.zeros(iperiod)

	# Create and diag the Floquet operator
	fo=CATFloquetOperator(grid,pot)

	# Create a coherent state localized in x0 with width = 2.0 in x
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)
		
	ncellini=1
	ncell2=int((ncellini-1)*0.5)
	for i in range(1,ncell2+1):
		xi=x0+i*2*np.pi
		wfi=WaveFunction(grid)
		wfi.setState("coherent",x0=xi,xratio=2.0)
		wf=wf+wfi
		xi=x0-i*2*np.pi
		wfi=WaveFunction(grid)
		wfi.setState("coherent",x0=xi,xratio=2.0)
		wf=wf+wfi

	wf.normalizeX()
	wf.x2p()
	
	ax=plt.gca()
	ax.plot(grid.x,np.abs(wf.x)**2,c="blue")
	xmax=2*ncellini*2*np.pi
	ax.set_xticks(np.linspace(-xmax/2,xmax/2,np.ceil(0.5*xmax/np.pi)+1,endpoint=True))
	ax.set_xticklabels([])
	ax.grid(which='major', color="red",alpha=1.0)	
	ax.set_xlim(-xmax/2,xmax/2)
	
	wf0=WaveFunction(grid)
	wf0.setState("coherent",x0=x0,xratio=2.0)
	xt=np.zeros(Ncell)
	
	for i in range(1,int((Ncell+1)/2)):
		
		xi=x0+i*2*np.pi
		xt[int((Ncell-1)/2+i)]=xi
		wfi=WaveFunction(grid)
		wfi.setState("coherent",x0=xi,xratio=2.0)
		wf0=wf0+wfi
		xi=x0-i*2*np.pi
		xt[int((Ncell-1)/2-i)]=xi
		wfi=WaveFunction(grid)
		wfi.setState("coherent",x0=xi,xratio=2.0)
		wf0=wf0+wfi

	wf0.normalizeX()
	wf0.x2p()
	
	
	

	for it in range(0,iperiod):
		fo.propagate(wf)
		xstd[it]=wf.getxstd()
		
		# ~ xpop=np.sum(np.reshape(wf.x*np.conjugate(wf0.x),(Ncell,128))*grid.ddx,axis=1)
		
		# ~ xstd=np.std(xpop)
		
		# ~ xstd[it]=np.sqrt(np.sum(xpop*xt**2))
		
		# ~ xstd[it]=np.sqrt(np.sum(wf.x*np.conjugate(wf0.x)*grid.x**2*grid.ddx))
			
		
			
		popNthcell[it]=wf.getxM(-np.pi+(Ncell-1)*np.pi,np.pi+(Ncell-1)*np.pi)
		
	ax.plot(grid.x,np.abs(wf.x)**2,c="red")
	plt.savefig(wdir+"wf/"+strint(runid))	
		
	# Save data
	np.savez(wdir+"runs/"+str(runid),"w", h=h, xstd=xstd,popNthcell=popNthcell,time=time,x=grid.x,wfx=wf.x)

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
		popNthcell=data['popNthcell']
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
		ax.plot(time,popNthcell)
		plt.savefig(wdir+"pics/"+strint(i))
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
	

