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
	os.mkdir(wdir+"trajectories")
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
	h=1.0/(np.linspace(1.0/hmax,1.0/hmin,nh)[runid])
	
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

	np.savez(wdir+"trajectories/"+strint(runid),"w", xL = xL, xR=xR)


	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	ax=plt.gca()
	ax.set_xlim(0,200)
	ax.set_ylim(0,1.0)
	s,nu,x0exp=convert2exp(gamma,h,x0)
	ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f} \quad h={:.3f} \quad 1/h={:.3f} \quad x0={:.1f}$".format(e,gamma,h,1/h,x0)+"\n"+r"$\varepsilon={:.3f} \quad s={:.3f} \quad \nu={:.3f} kHz \quad x_0={:.1f}^o$".format(e,s,nu/10**3,x0exp))
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

	iTF=int(iperiod)
	

	#density=np.zeros((int(iperiod/2)+1,nh))
	
	
	hm=np.linspace(1.0/hmax,1.0/hmin,nh)
	s=np.linspace(1.0/hmax,1.0/hmin,nh)**2*4*gamma
	omegas=np.fft.rfftfreq(iTF,d=2.0)#*2*np.pi

	X,Y=np.meshgrid(hm,omegas)
	
	Z=np.zeros((int(iTF/2)+1,nh))
	print(X.shape,Y.shape,Z.shape)
	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	

	for ih in range(0,nh):
		data=np.load(wdir+"trajectories/"+strint(ih)+".npz")
		xL=data['xL']*np.exp(-time/1000)
		xR=data['xR']*np.exp(-time/1000)
		xLf=np.abs(np.fft.rfft(xL))
		xRf=np.abs(np.fft.rfft(xR))
		xLf[0]=0.0
		xRf[0]=0.0
		

		Z[:,ih]=(xLf+xRf)*0.5

	np.savez(wdir+"nu-heff","w", hm=X,omegas=Y,tf=Z,e=e,gamma=gamma,x0=x0,iperiod=iperiod)
	ax=plt.gca()

	ax.set_xlabel("1/heff")
	ax.set_ylabel("frequence")
	ax.set_title(r"$\varepsilon={:.3f} \quad \gamma={:.3f} \quad x_0={:.1f} $".format(e,gamma,x0))
	ax.set_yscale("log")
	omegamax=max(omegas)
	omegamin=0.001
	ax.set_ylim(omegamin,omegamax)

	cmap = plt.get_cmap('Greys')
	
	
	
	plt.pcolormesh(X,Y,Z,cmap=cmap)
	
	convert2theory(0.1,0.2)
	
	plt.errorbar(1/0.3959,1/25.6,yerr=0.5/128,fmt='o')
	
	plt.errorbar(1/0.3838,0.02344,yerr=0.5/128,fmt='o')
	
	# ~ plt.errorbar(2.77,1/55.0,yerr=0.5/128,fmt='o')
	
	# ~ plt.errorbar(2.87,1/90.0,yerr=0.5/128,fmt='o')
	
	# ~ plt.errorbar(2.97,1/37.5,yerr=0.5/128,fmt='o')
	
	# ~ plt.errorbar(1/0.346,1/110.0,yerr=0.5/128,fmt='o')
	# ~ plt.errorbar(1/0.346,1/20.0,yerr=0.5/128,fmt='o')
	
	# ~ plt.errorbar(2.563,1/19.0,yerr=1/100,fmt='o')
	# ~ plt.errorbar(3.361,1/30.0,yerr=1/100,fmt='o')
	# ~ plt.errorbar(3.398,1/40.0,yerr=1/100,fmt='o')
	# ~ plt.errorbar(4.999,1/50.0,yerr=1/130,fmt='o')
	
	# ~ for x0 in [2.57,2.67,2.77,2.87,2.97,3.07,3.17,3.27,3.37]:
		# ~ plt.plot([x0,x0],[omegamin,omegamax])
	# ~ plt.plot([2.887,2.887],[omegamin,omegamax])
	#plt.savefig(wdir+"final-heff.pdf",bbox_inches = 'tight',format="pdf") 
	
	# ~ X,Y=np.meshgrid(s,omegas)
	# ~ plt.pcolormesh(X,Y,Z,cmap=cmap)
	# ~ #plt.show()
	plt.savefig(wdir+"final-s.png",bbox_inches = 'tight',format="png",dpi=150) 

	
	


