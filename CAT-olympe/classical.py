import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.doublewell import *
from utils.systems.kickedrotor import *
from utils.systems.general import *
from utils.toolsbox import *
from utils.mathtools.periodicfunctions import *
from utils.plot import *
from matplotlib import patches
from matplotlib.path import Path

# ~ sys.path.insert(0, '/users/martinez/Documents/Cloud CNRS/Plot/')
# ~ import latex


# State: stable [22/02/2019]

# To be used with "run-classical.slurm"

# This scripts makes possibles to 
# 1. generate a stroboscopic trajectory
# 2. export the SPS (.png)

# Arguments to provide:
# 1. mode = "compute", "plot"
# 2. working directory
# if mode=="compute":
# 	3. input file
# 	4. total number of tasks
# 	5. id of the current runs

mode=sys.argv[1] # mode selected
wdir=sys.argv[2] # working (=output) directory

if mode=="initialize":
	os.mkdir(wdir)
	os.mkdir(wdir+"trajectories")

if mode=="compute":
	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	# Description of the run
	description=data['description']

	# General physical parameters 
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	s,nu,x0exp=convert2exp(gamma,h)
	iperiod=data['iperiod']
	iperiod=10000
	phase=1.75*np.pi
	X0=-1
	Y0=1.17

	data.close() # close file
	
	
	
	# run info
	ny0=int(sys.argv[4]) # total number of runs
	runid=int(sys.argv[5])-1 # ID of the current run
	if runid==0: # This generate and parameters files in the working directory with read input (to avoid surprises)		
		np.savez(wdir+"params","w", description=description, ny0=ny0, e=e,gamma=gamma,h=h,iperiod=iperiod,s=s,nu=nu,phase=phase,X0=X0,Y0=Y0)

	# Create the potential, time propagator and stroboscopic phase space
	
	mod=PeriodicFunctions(phase=phase)
	pot=PotentialMP(e,gamma,f=mod.cos)
	# ~ pot=PotentialDW(e,gamma)
	cp=ClassicalContinueTimePropagator(pot)
	pp=PhasePortrait(iperiod,ny0,cp,xmax=np.pi,pmax=2.5,theta1=pot.thetaR1(),theta2=pot.thetaR2(),X0=X0,Y0=Y0) 
	

	# Generate and save a trajectory
	x,p=pp.getTrajectory(runid)
	sc=pp.getChaoticity(x,p)
	np.savez(wdir+"trajectories/"+str(runid),"w", x=x, p=p,sc=sc)
	
if mode=="gather":
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	h=data['h']	
	iperiod=data['iperiod']
	ny0=data['ny0']
	data.close()
	
	sc=np.zeros((ny0,iperiod))
	x=np.zeros((ny0,iperiod))
	p=np.zeros((ny0,iperiod))
	norms=0.0
	for i in range(0,ny0):
		data=np.load(wdir+"trajectories/"+str(i)+".npz")
		x[i]=data['x']
		p[i] = data['p']
		sc[i]= data['sc']*np.ones(iperiod)
		data.close()
	np.savez(wdir+"all-trajectories","w", x=x, p=p,sc=sc/np.max(sc))

if mode=="plot":
	# Loading inpute file
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	iperiod=int(data['iperiod'])
	phase=data['phase']
	X0=data['X0']
	Y0=data['Y0']
	ny0=data['ny0']
	data.close()
	
	# General plotting setup
	fig, ax = plt.subplots(figsize=(np.pi*2,4),frameon=False)
	ax.set_xlim(-np.pi,np.pi)
	ax.set_ylim(-2.0,2.0)
	ax.set_xticks([])
	ax.set_yticks([])
	
	# ~ ax.set_aspect('equal')
	# ~ ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f}$".format(e,gamma))
	# ~ ax.set_xlabel(r"$x$")
	# ~ ax.set_ylabel(r"$p$")
	

	# Plotting the SPS
	cmap=plt.get_cmap("jet")

	data=np.load(wdir+"all-trajectories.npz")
	# ~ x=data["x"].flatten()
	# ~ p=data["p"].flatten()
	# ~ sc=data["sc"].flatten()
	x=data["x"]
	p=data["p"]
	sc=data["sc"]
	sc=sc/np.max(sc)
	print(x.shape)
	data.close()
	
	
	mod=PeriodicFunctions(phase=phase)
	pot=PotentialMP(e,gamma,f=mod.cos)
	cp=ClassicalContinueTimePropagator(pot)
	pp=PhasePortrait(iperiod,ny0,cp,xmax=np.pi,pmax=2.5,theta1=pot.thetaR1(),theta2=pot.thetaR2(),X0=X0,Y0=Y0)
	
	plt.scatter(pot.R1()*np.cos(pot.thetaR1()),pot.R1()*np.sin(pot.thetaR1()))
	# ~ plt.scatter(pot.R1()*np.cos(pot.thetaR2()),pot.R1()*np.sin(pot.thetaR2()))
	
	# ~ for i in range(0,ny0):
		# ~ y=pp.generatey0(i)
		# ~ plt.scatter(y[0],y[1],c="red")
	
	
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	
	# ~ plt.scatter(x,p,c=plt.cm.get_cmap('bwr')(sc),s=0.015**2)
	# ~ plt.scatter(x,p,c="blue",s=0.015**2)
	
	cond=sc>0.5	
	xchaotic=np.extract(cond,x)
	pchaotic=np.extract(cond,p)
	xreg=np.extract(1-cond,x)
	preg=np.extract(1-cond,p)
	
	plt.scatter(xchaotic,pchaotic,s=0.02**2,c="red")
	plt.scatter(xreg,preg,s=0.02**2,c="blue")
	
	# ~ plt.scatter(xchaotic,pchaotic,s=0.1**2,c="red")
	# ~ plt.scatter(xreg,preg,s=0.1**2,c="blue")
	
	# ~ plt.show()
		

	plt.savefig(wdir+"phase-portrait2.png",dpi=500)
	# ~ plt.savefig(wdir+"phase-portrait.eps", bbox_inches = 'tight',format="eps")
	
	
if mode=="plot2":
	# Loading inpute file
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	ny0=data['ny0']
	data.close()
	
	# General plotting setup
	fig = latex.fig(columnwidth=345.0,wf=1.0)
	ax = plt.gca()
	ax.set_xlim(-np.pi,np.pi)
	ax.set_ylim(-2.0,2.0)
	ax.set_aspect('equal')
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$p$")
	# ~ ax.set_xticks([])
	# ~ ax.set_yticks([])
	
	
	props = dict(boxstyle='square', facecolor='white', alpha=1.0)
	prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.5",shrinkA=0,shrinkB=0,fc="black",lw=1.2)

	ax.annotate("", xy=(0.95, 0.2), xytext=(0.825, 0.92), arrowprops=prop,xycoords='axes fraction')
	ax.annotate("", xy=(0.70, 0.57), xytext=(0.825, 0.92), arrowprops=prop,xycoords='axes fraction')
	ax.annotate("", xy=(0.52, 0.81), xytext=(0.825, 0.92), arrowprops=prop,xycoords='axes fraction')
	ax.text(0.825, 0.92, r'Regular orbits', transform=ax.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center', bbox=props)

	ax.annotate("", xy=(0.52, 0.72), xytext=(0.15, 0.92), arrowprops=prop,xycoords='axes fraction')
	ax.annotate("", xy=(0.1, 0.5), xytext=(0.15, 0.92), arrowprops=prop,xycoords='axes fraction')
	ax.text(0.15, 0.92, r'Chaotic sea', transform=ax.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center', bbox=props)
	
	ax.annotate("", xy=(0.63, 0.5), xytext=(0.2, 0.08), arrowprops=prop,xycoords='axes fraction')
	ax.annotate("", xy=(1-0.63, 0.5), xytext=(0.2, 0.08), arrowprops=prop,xycoords='axes fraction')
	ax.text(0.2, 0.08, r'$(x \rightarrow -x)$ islands', transform=ax.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center', bbox=props)
	
	ax.annotate("", xy=(0.62, 0.81), xytext=(0.8, 0.08), arrowprops=prop,xycoords='axes fraction')
	ax.annotate("", xy=(0.62, 1-0.81), xytext=(0.8, 0.08), arrowprops=prop,xycoords='axes fraction')
	ax.text(0.8, 0.08, r'$(p \rightarrow -p)$ islands', transform=ax.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center', bbox=props)
	
	# Plotting the SPS
	cmap=plt.get_cmap("jet")

	data=np.load(wdir+"all-trajectories.npz")
	x=data["x"]
	p=data["p"]
	sc=data["sc"]
	sc=sc/np.max(sc)
	print(x.shape)
	data.close()
	
	
	cond=sc>0.5	
	xchaotic=np.extract(cond,x)
	pchaotic=np.extract(cond,p)
	xreg=np.extract(1-cond,x)
	preg=np.extract(1-cond,p)
	
	
	plt.scatter(xchaotic,pchaotic,s=0.01**2,c="red")
	plt.scatter(xreg,preg,s=0.01**2,c="blue")

	latex.save(wdir+"phase-portrait",form="png")
	
if mode=="plot3":
	# Loading inpute file
	data=np.load(wdir+"params.npz")
	e=data['e']
	gamma=data['gamma']
	ny0=data['ny0']
	data.close()
	
	# General plotting setup
	fig = latex.fig(columnwidth=345.0,wf=1.0)
	ax = plt.gca()
	ax.set_xlim(-np.pi,np.pi)
	ax.set_ylim(-2.0,2.0)
	# ~ ax.set_ylim(-np.pi,np.pi)
	
	ax.set_aspect('equal')
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$p$")
	ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
	ax.set_xticklabels([r"-$\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
	ax.set_yticks([-np.pi/2,0,np.pi/2])
	ax.set_yticklabels([r"$-\pi/2$",r"$0$",r"$\pi/2$"])
	
	
	props = dict(boxstyle='square', facecolor='white', alpha=1.0)
	prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.5",shrinkA=0,shrinkB=0,fc="black",lw=1.2)

	ax.set_title(r"$t/T=0.875$")
	
	# Plotting the SPS
	cmap=plt.get_cmap("jet")

	data=np.load(wdir+"all-trajectories.npz")
	x=data["x"]
	p=data["p"]
	sc=data["sc"]
	sc=sc/np.max(sc)
	print(x.shape)
	data.close()
	
	
	cond=sc>0.5	
	xchaotic=np.extract(cond,x)
	pchaotic=np.extract(cond,p)
	xreg=np.extract(1-cond,x)
	preg=np.extract(1-cond,p)
	
	
	plt.scatter(xchaotic,pchaotic,s=0.01**2,c="red")
	plt.scatter(xreg,preg,s=0.01**2,c="blue")

	latex.save(wdir+"phase-portrait",form="png")
