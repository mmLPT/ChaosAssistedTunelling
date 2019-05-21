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

# This scripts makes possibles to 
# 1. compute in // the free propagation for different quasi-momentum
# 2. gather and average the results
# 3. plot the averaged data
 
mode=sys.argv[1]
wdir=sys.argv[2]

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"pictures")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	# Description of the run
	description=data['description']

	# General physical parameters 
	N=int(data['N'])
	e=data['e']
	x0=data['x0']
	gamma=data['gamma']

	# Free propagation
	iperiod=int(data['iperiod']) #number of period

	# heff values
	hmin=data['hmin']
	hmax=data['hmax']

	data.close() # close the input file
	
	# Initialization of potential and correcting the x0 value if needed
	pot=PotentialMP(e,gamma)
	if x0==0.0:
		x0=pot.x0
	
	nh=int(sys.argv[4]) # number of runs for a given h
	runid=int(sys.argv[5])-1 # Id of the current run

	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description,N=N, e=e,gamma=gamma,x0=x0,hmin=hmin,hmax=hmax,nh=nh,iperiod=iperiod)

	# Create array to store "Left" and "Right" observables
	xR=np.zeros(iperiod)
	xL=np.zeros(iperiod)

	# Initialization of the grid for given h value
	h=np.linspace(hmin,hmax,nh)[runid]
	grid=Grid(N,h)
	
	# Create the Floquet operator
	fo=CATFloquetOperator(grid,pot)

	# Create the initial state: a coherent state localized in x0 with width = 2.0 in x
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=x0,xratio=2.0)

	# Propagate the wavefunction over iperiod storing the observable every time
	for i in range(0,iperiod):
		xL[i]=wf.getxL()
		xR[i]=wf.getxR()
		fo.propagate(wf)

	A=max(xR[0],xL[0])
	xL=xL/A
	xR=xR/A

	np.savez(wdir+str(runid),"w", xL = xL, xR=xR)


	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	ax=plt.gca()
	ax.set_xlim(0,200)
	ax.set_ylim(0,1.0)
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.2f} \quad h={:.3f}$".format(e,gamma,h))
	plt.plot(time,xL, c="red")
	plt.plot(time,xR, c="blue")
	plt.savefig(wdir+"pictures/"+strint(runid)+".png") 


if mode=="final":
	data=np.load(wdir+"params.npz")
	nh=data['nh']
	iperiod=data['iperiod']
	hmin=data['hmin']
	hmax=data['hmax']
	e=data['e']
	gamma=data['gamma']
	x0=data['x0']
	data.close()

	iTF=100 
	#iTF=int(iperiod)

	#density=np.zeros((int(iperiod/2)+1,nh))
	density=np.zeros((int(iTF/2)+1,nh))
	h=np.linspace(hmin,hmax,nh)
	print(nh)
	omegas=np.fft.rfftfreq(iTF,d=2.0)*2*np.pi

	a=0
	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	for ih in range(0,nh):
		data=np.load(wdir+str(ih)+".npz")
		xL=data['xL'][:iTF].copy()
		xR=data['xR'][:iTF].copy()
		#plt.plot(xL)
		#plt.show()
		xLf=np.abs(np.fft.rfft(xL))
		xRf=np.abs(np.fft.rfft(xR))
		xLf[0]=0.0
		xRf[0]=0.0
		density[:,ih]=(xLf+xRf)*0.5
		a=max(a,max(density[:,ih]))

		#time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
		#ax=plt.gca()
		#ax.set_xlim(0,1000)
		#ax.set_ylim(0,1.0)
		#ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.2f} \quad h={:.3f}$".format(e,gamma,h[ih]))
		#plt.plot(time,data['xL'], c="red")
		#plt.plot(time,data['xR'], c="blue")
		#plt.savefig(wdir+"pictures/"+strint(ih)+".png") 
		#plt.clf()

	a=np.percentile(density,99.80)
	density=density/a

	ax=plt.gca()
	#ax.set_xlim(min(h),max(h))
	#ax.set_ylim(0.001,max(omegas))
	ax.set_xlabel("h")
	ax.set_ylabel("omega")
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f}$".format(e,gamma))
	omegas[0]=omegas[1]
	#ax.set_ylim(min(1.0/omegas),max(1.0/omegas)/2.0)
	#ax.set_ylim(min(omegas),max(omegas))
	#ax.set_yscale("log")
	levels = MaxNLocator(nbins=100).tick_values(0.0,1.0)	
	cmap = plt.get_cmap('Greys')
	norm = colors.LogNorm(0.01,1.0) 
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
	#plt.contourf(h,omegas,np.sqrt(density), levels=levels,cmap=cmap)
	plt.pcolormesh(h,omegas,density, norm=norm,cmap=cmap)
	#plt.pcolormesh(h,1/omegas,np.sqrt(density), norm=norm,cmap=cmap)
	plt.savefig(wdir+"final.png")
	#plt.show()

if mode=="checkTF":
	data=np.load(wdir+"params.npz")
	nh=data['nh']
	iperiod=data['iperiod']
	hmin=data['hmin']
	hmax=data['hmax']
	e=data['e']
	gamma=data['gamma']
	x0=data['x0']
	data.close()

	iTF=100
	h0=0.245
	dh=abs((hmax-hmin)/nh)
	ih=int((h0-hmin)/dh)
	omegas=np.fft.rfftfreq(int(iTF),d=2.0)*2*np.pi

	h=np.linspace(hmin,hmax,nh)[ih]


	data=np.load(wdir+str(ih)+"/averaged.npz")
	time=data['time']
	xL=data['xL'][:iTF].copy()
	xR=data['xR'][:iTF].copy()
	
	xLf=np.abs(np.fft.rfft(xL))
	xRf=np.abs(np.fft.rfft(xR))
	xLf[0]=0.0
	xRf[0]=0.0

	#setLatex()
	f = plt.figure()
	ax = f.add_subplot(3, 1, 1)
	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f} \quad h={:.3f}$".format(e,gamma,h))
	ax.set_xlim(0,iTF*2.0)
	plt.plot(time,data['xL'])
	plt.plot(time,data['xR'])
	ax = f.add_subplot(3, 1, 2)
	plt.scatter(omegas,xLf)
	ax = f.add_subplot(3, 1, 3)
	omegas[0]=omegas[int(iTF/2)-1]
	ax.set_xscale('log')
	plt.scatter(2*np.pi/omegas,xLf)
	plt.savefig(wdir+"TF+freepop-h{:.3f}.png".format(h))
	#plt.show()

	


