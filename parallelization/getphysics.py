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

if mode=="compute":

	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	description=data['description']  

	N=int(data['N'])
	e=data['e']
	gamma=data['gamma']
	x0=data['x0'] 

	hmin=data['hmin']
	hmax=data['hmax']

	iperiod=100 #int(data['iperiod'])
	nstates=int(data['nstates'])

	data.close()

	# Initialization of potential and correcting the x0 value if needed
	
	nh=int(sys.argv[4]) # number of runs for a given h
	runid=int(sys.argv[5])-1 # Id of the current run

	if runid==0:	
		np.savez(wdir+"params","w", description=description,N=N, e=e,gamma=gamma,x0=x0,hmax=hmax,hmin=hmin,iperiod=iperiod,nstates=nstates,nh=nh)
		os.mkdir(wdir+"pictures")

	os.mkdir(wdir+str(runid))

	h=np.linspace(hmin,hmax,nh)[runid]

	# Initialization of the grid for given h value
	grid=Grid(N,h)
	husimi=Husimi(grid)
	pot=PotentialMP(e,gamma)

	# Creating array to store data
	qEs=np.zeros(nstates)
	overlaps=np.zeros(nstates)
	symX=np.zeros(nstates)
	ind=np.zeros(nstates)

	# Create and diag the Floquet operator
	fo=CATFloquetOperator(grid,pot)
	fo.diagonalize()

	# Create a coherent state localized in x0 with width = 2.0 in x
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=x0,xratio=2.0)

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	fo.computeOverlapsAndQEs(wfcs)

	# Get the quasienergies, overlaps and symetries of the nstates with highest overlap on wfcs
	qEs,overlaps,symX,ind=fo.getQEsOverlapsSymmetry(nstates,indexbool=True)

	for i in range(0,nstates):
		husimi.save(fo.getEvec(ind[i],twolower=False),wdir+str(runid)+"/"+strint(i),convert=False)

	xR=np.zeros(iperiod)
	xL=np.zeros(iperiod)

	print(symX)

	# Propagate the wavefunction over iperiod storing the observable every time
	for i in range(0,iperiod):
		xL[i]=wfcs.getxL()
		xR[i]=wfcs.getxR()
		fo.propagate(wfcs)

	A=max(xR[0],xL[0])
	xL=xL/A
	xR=xR/A
	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	omegas=np.fft.rfftfreq(iperiod,d=2.0)*2*np.pi
	xLf=np.abs(np.fft.rfft(xL))
	xRf=np.abs(np.fft.rfft(xR))
	xLf[0]=0
	xRf[0]=0

	f = plt.figure()
	
	for i in range(0,nstates):
		ax = f.add_subplot(4, nstates, 2*nstates+int(i+1))
		data=np.load(wdir+str(runid)+"/"+strint(i)+".npz")
		rho=data["rho"]
		x=data["x"]
		p=data["p"]
		R=rho.max()
		rho=rho/R
		data.close()
		
		# Generla settings : tile/axes
		ax.set_ylim(-np.pi,np.pi)
		ax.set_xlim(-np.pi,np.pi)
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

		# 2D map options
		levels = MaxNLocator(nbins=150).tick_values(0.0,1.0)	
		cmap = plt.get_cmap('plasma')
		norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
		plt.contourf(x,p,rho, levels=levels,cmap=cmap)

	omegasth=np.zeros(int(np.math.factorial(nstates)/(np.math.factorial(nstates-2)*2)))
	ovver=np.zeros(int(np.math.factorial(nstates)/(np.math.factorial(nstates-2)*2)))
	k=0

	print(int(np.math.factorial(nstates)/(np.math.factorial(nstates-2)*2)))
	for i in range(0,nstates-1):
		for j in range(i+1,nstates):
			print(i,j)
			omegasth[k]=2*np.pi/h*abs(fo.diffqE1qE2(ind[i],ind[j]))
			ovver[k]=(overlaps[i]*overlaps[j])
			k=k+1

	ax = f.add_subplot(4, 1, 4)
	plt.plot(time,xL, c="red")
	plt.plot(time,xR, c="blue")
	ax.set_xlim(0,max(time))
	ax.set_ylim(0,1.0)


	ax = f.add_subplot(2, 1, 1)

	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.2f} \quad h={:.3f}$".format(e,gamma,h))
	ax.set_xlim(0,max(omegas))
	plt.plot(omegas,xLf, c="red")
	plt.plot(omegas,xRf, c="blue")
	ax2 = ax.twinx()
	ax2.set_ylim(0,1.0)
	k=0
	for i in range(0,nstates-1):
		for j in range(i+1,nstates):
			#plt.text(omegasth[i],ovver[i], str(i)+"-"+str(j), fontsize=12)
			if not(symX[i]==symX[j]):
				plt.scatter(omegasth[k],ovver[k],label=str(i)+"-"+str(j)+"-{:.1f}".format(omegasth[k]))
			k=k+1
	ax2.legend(loc='upper center')
	# Saving fig
	plt.savefig(wdir+"pictures/"+strint(runid)+".png", bbox_inches='tight')

if mode=="solorun":

	# This mode compute the spectrum for a single value of h
	# It is made to be proceed on a single process
	if not os.path.exists(wdir):
		os.mkdir(wdir)

	# Loading input file
	inputfile=sys.argv[3]
	data=np.load(inputfile+".npz")
	
	description=data['description']  

	N=int(data['N'])
	e=data['e']
	gamma=data['gamma']
	h=data['h']
	x0=data['x0'] 

	iperiod=250 #int(data['iperiod'])
	nstates=3 #int(data['nstates'])

	data.close()

	# Initialization of the grid for given h value
	grid=Grid(N,h)
	husimi=Husimi(grid)
	pot=PotentialMP(e,gamma)

	# Creating array to store data
	qEs=np.zeros(nstates)
	overlaps=np.zeros(nstates)
	symX=np.zeros(nstates)
	ind=np.zeros(nstates)

	# Create and diag the Floquet operator
	fo=CATFloquetOperator(grid,pot)
	fo.diagonalize()

	# Create a coherent state localized in x0 with width = 2.0 in x
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=x0,xratio=2.0)

	# Compute the overlap between wfcs and the eigenstates of the Floquet operator
	fo.computeOverlapsAndQEs(wfcs)

	# Get the quasienergies, overlaps and symetries of the nstates with highest overlap on wfcs
	qEs,overlaps,symX,ind=fo.getQEsOverlapsSymmetry(nstates,indexbool=True)

	for i in range(0,nstates):
		husimi.save(fo.getEvec(ind[i],twolower=False),wdir+strint(i),convert=False)

	xR=np.zeros(iperiod)
	xL=np.zeros(iperiod)

	print(symX)

	# Propagate the wavefunction over iperiod storing the observable every time
	for i in range(0,iperiod):
		xL[i]=wfcs.getxL()
		xR[i]=wfcs.getxR()
		fo.propagate(wfcs)

	A=max(xR[0],xL[0])
	xL=xL/A
	xR=xR/A
	time=2.0*np.linspace(0.0,1.0*iperiod,num=iperiod,endpoint=False)
	omegas=np.fft.rfftfreq(iperiod,d=2.0)*2*np.pi
	xLf=np.abs(np.fft.rfft(xL))
	xRf=np.abs(np.fft.rfft(xR))
	xLf[0]=0
	xRf[0]=0

	f = plt.figure()
	
	for i in range(0,nstates):
		ax = f.add_subplot(2, nstates,  nstates+int(i+1))
		data=np.load(wdir+strint(i)+".npz")
		rho=data["rho"]
		x=data["x"]
		p=data["p"]
		R=rho.max()
		rho=rho/R
		data.close()
		
		# Generla settings : tile/axes
		ax.set_ylim(-np.pi,np.pi)
		ax.set_xlim(-np.pi,np.pi)
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

		# 2D map options
		levels = MaxNLocator(nbins=150).tick_values(0.0,1.0)	
		cmap = plt.get_cmap('plasma')
		norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
		plt.contourf(x,p,rho, levels=levels,cmap=cmap)
		ax.text(-3, 2, str(i),fontsize = 15, color = 'black', backgroundcolor = 'white')

	omegasth=np.zeros(int(np.math.factorial(nstates)/(np.math.factorial(nstates-2)*2)))
	ovver=np.zeros(int(np.math.factorial(nstates)/(np.math.factorial(nstates-2)*2)))
	k=0

	print(int(np.math.factorial(nstates)/(np.math.factorial(nstates-2)*2)))
	for i in range(0,nstates-1):
		for j in range(i+1,nstates):
			print(i,j)
			omegasth[k]=2*np.pi/h*abs(fo.diffqE1qE2(ind[i],ind[j]))
			ovver[k]=(overlaps[i]*overlaps[j])
			k=k+1

	#ax = f.add_subplot(3, 1, 3)
	#plt.plot(time,xL, c="red")
	#plt.plot(time,xR, c="blue")
	#ax.set_xlim(0,max(time))
	#ax.set_ylim(0,1.0)


	ax = f.add_subplot(2, 1, 1)

	ax.set_title(r"$\varepsilon={:.2f} \quad \gamma={:.3f} \quad h={:.3f}$".format(e,gamma,h))
	ax.set_xlim(0,max(omegas))
	plt.plot(omegas,xLf, c="red",zorder=1)
	#plt.plot(omegas,xRf, c="blue")
	#ax2 = ax.twinx()
	#ax2.set_ylim(0,1.0)
	k=0
	norma=max(ovver)/max(xRf)
	for i in range(0,nstates-1):
		for j in range(i+1,nstates):

			if not(symX[i]==symX[j]):
				plt.scatter(omegasth[k],ovver[k]/norma,label=str(i)+" avec "+str(j),zorder=2)
			k=k+1
	ax.legend(loc='upper center')
	# Saving fig
	plt.savefig(wdir+"final.png", bbox_inches='tight')	

