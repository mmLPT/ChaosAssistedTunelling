import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
#from utils.mathtools.periodicfunctions import *
import utils.plot.read as read
import modesbasic

# The following modes make possible to study the influence of the confinement
# in the x direction. The study are made on a single celle, with a shifted
# harmonic potential: V(x)=1/2*omega^2*(x-x1)^2 
# The nondimenzionalization gives omega =(grid.h*25.0)/(2*8113.9)
# The way the potential class is build, forces us to gives this information
# to the constructor since it depend on heff (that usually vary).

			
def perturbation_theory(grid,e,gamma,datafile="data/perturbation", compute=False, read=True):
	
	# This mode is used to study the prediction of the perturbation theory
	# on tunneling period and quasi-energies, versus the computed one.
	# The floquet operator is diagonalized for different value of shift
	# (for the harmonic perturbation)
	
	if(compute):
		# Grid for x1 (shift of harmonic perturbation)
		ncellmax=20
		npoints=31
		x1=np.linspace(-ncellmax*2*np.pi,ncellmax*2*np.pi,npoints)
		
		# Data array
		T=np.zeros(npoints) # Period from floquet diagonalization
		Tth=np.zeros(npoints) # Period from perturbation theory
		qE=np.zeros((npoints,2)) # Quasi-energies from floquet diagonalization
		qEth=np.zeros((npoints,2))# Quasi-energies from perturbation  theory
		
		# Unperturbed harmonic potential
		pot0=PotentialMP(e,gamma)
		
		# Coherent states localized on left (m) and right (p)
		wfcsp=WaveFunction(grid)
		wfcsp.setState("coherent",x0=pot0.x0,xratio=2.0)
		wfcsm=WaveFunction(grid)
		wfcsm.setState("coherent",x0=-pot0.x0,xratio=2.0)
		
		# Findind initial tunneling states that will be used to compute
		# perturbation theory expectation later
		fo0=CATFloquetOperator(grid,pot0,T0=4*np.pi,idtmax=500)
		fo0.diagonalize()
		fo0.findTunellingStates(wfcsp)

		for i in range(0,npoints):
			
			# Creation of an operator, based on an asymetric potential
			pot=PotentialMPasym(e,gamma,x1[i],(grid.h*25.0)/(2*8113.9))
			fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
			
			# Determine the tunneling states
			fo.diagonalize()
			if x1[i]>0.0: # This was used for large value of perturbation to avoid non-symetric plot
				fo.findTunellingStates(wfcsp)
			else:
				fo.findTunellingStates(wfcsm)
				
			# Get the tunneling period and the quasi-energies for the
			# 2 levels that are tunneling
			T[i]=fo.getTunnelingPeriod()
			qE[i]=fo.getQE(0),fo.getQE(1)

			# Get expected period/quasi-energies from perturbation theory
			qEth[i]=fo0.getQETh(0,pot),fo0.getQETh(1,pot)
			Tth[i]=0.5*grid.h/(qEth[i][1]-qEth[i][0])

			# Console output
			print("x=",x1[i],(qE[i][0]-qEth[i][0])/qE[i][0],(qE[i][1]-qEth[i][1])/qE[i][1])
			print("T-Tth/Tth",abs(Tth[i]-T[i])/Tth[i])
			
		# Saving in a datafile
		np.savez(datafile,"w", x1=x1, T=T, Tth=Tth,qE=qE,qEth=qEth)
		
	if(read):
		# Routine to plot
		
		data=np.load("data/perturbation.npz")
		x1=data['x1']
		T=data['T']
		Tth=data['Tth']
		qE=data['qE']
		qEth=data['qEth']

		plt.plot(x1,qE[:,0],c="blue")
		plt.scatter(x1,qEth[:,0],c="green")
		plt.plot(x1,qEth[:,1],c="red")
		plt.scatter(x1,qE[:,1],c="yellow")

		plt.show()
		plt.plot(x1,abs(Tth-T)/Tth,c="red")
		plt.show()

def check_T_with_confinment(imax=11, e=0.32, gamma=0.29,N=64, xasym=15*2*np.pi, datafile="data/h_with_asym",compute=False,read=True):
	
	# This mode is used to compare the value of
	
	if(compute):
		h=np.linspace(0.19,0.41,imax)
		
		T_x0=np.zeros(imax)
		T_xasym=np.zeros(imax)

		for i in range(0,imax):
			# Initialization
			grid=Grid(N,h[i],2*np.pi)
			
			pot_x0=PotentialMPasym(e,gamma,0,(grid.h*25.0)/(2*8113.9))
			pot_xasym=PotentialMPasym(e,gamma,xasym,(grid.h*25.0)/(2*8113.9))
			
			wfcs=WaveFunction(grid)
			wfcs.setState("coherent",x0=pot_xasym.x0,xratio=2.0)
			
			# Harmonic potential at x=0
			fo_x0=CATFloquetOperator(grid,pot_x0,T0=4*np.pi,idtmax=500)
			fo_x0.diagonalize()
			fo_x0.findTunellingStates(wfcs)
			T_x0[i]=fo_x0.getTunnelingPeriod()

			# Harmonic potential at x=xshift
			fo_xasym=CATFloquetOperator(grid,pot_xasym,T0=4*np.pi,idtmax=500)
			fo_xasym.diagonalize()
			fo_xasym.findTunellingStates(wfcs)
			T_xasym[i]=fo_xasym.getTunnelingPeriod()
			
			
			print(str(i+1),"/",imax,"h=",h[i])
			
		np.savez(datafile,"w", h=h, T_x0=T_x0, T_xasym=T_xasym)
		
	if(read):
		data=np.load(datafile+".npz")
		h=data['h']
		T_x0=data['T_x0']
		T_xasym=data['T_xasym']

		imax=h.shape[0]
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim(10**(-8),1.0)

		plt.scatter(T_x0,(T_xasym-T_x0),c="red")
		plt.scatter(T_x0,-(T_xasym-T_x0),c="blue")
		#plt.plot(h,T_xasym,c="red")
		plt.show()
	
def symetry_of_gs_with_h(N, e, gamma, datafile="split", compute=False,read=True):
	if(compute):
		imax=100
		#h=np.linspace(0.1,0.3,imax)
		h=np.linspace(0.203,0.212,imax)

		T=np.zeros(imax)
		Tasym=np.zeros(imax)
		qE=np.zeros((imax,2))
		n=10
		qEs=np.zeros((imax,n))
		isLowerSymetric=np.zeros(imax, dtype=bool)
		isPeriodGrowing=np.zeros(imax, dtype=bool)
		
		for i in range(0,imax):
			print(str(i+1)+"/"+str(imax)+" - h={0:.25f}".format(h[i]))
			grid=Grid(N,h[i],2*np.pi)
			pot=PotentialMP(e,gamma)
			potasym=PotentialMPasym(e,gamma,15*2*np.pi,(grid.h*25.0)/(2*8113.9))
				
			wfcs=WaveFunction(grid)
			wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
			
			fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=2500)
			fo.diagonalize()
			fo.findTunellingStates(wfcs)
			T[i]=fo.getTunnelingPeriod()
			qE[i]=fo.getQE(0),fo.getQE(1)
			qEs[i]=fo.getQEs(n)
			isLowerSymetric[i]=fo.getEvec(0).isSymetricInX()
			
			#~ foasym=CATFloquetOperator(grid,potasym,T0=4*np.pi,idtmax=500)
			#~ foasym.diagonalize()
			#~ foasym.findTunellingStates(wfcs)
			#~ Tasym[i]=foasym.getTunnelingPeriod()
			#~ isPeriodGrowing[i]=Tasym[i]>T[i]
			Tasym[i]=0
			isPeriodGrowing[i]=False
			
			#~ plt.plot(grid.x,np.real(fo.getEvec(0).x),c="blue",label="GS")
			#~ plt.plot(grid.x,np.real(fo.getEvec(1).x),c="red",label="1st ES")
			#~ plt.legend()
			#~ plt.show()
		
		np.savez(datafile,"w", h=h, T=T,qE=qE, qEs=qEs,isLowerSymetric=isLowerSymetric,isPeriodGrowing=isPeriodGrowing)
	
	if(read):
		data=np.load(datafile+".npz") # pas zoomé
		#data=np.load("data/croisement4.npz") # très zoomé

		fig, ax1 = plt.subplots()
		T=data['T']
		ax2=ax1.twinx()
		isLowerSymetric=data['isLowerSymetric']
		isPeriodGrowing=data['isPeriodGrowing']
		h=data['h']
		qE=data['qE']
		qEs=data['qEs']
		ax1.set_xlim(min(h),max(h))
		#ax1.set_ylim(min(qE[:,0]),max(qE[:,1]))
		print(qE[:,1]-qE[:,0])
		for i in range(0,6):
			ax1.scatter(h,qEs[:,i]-qEs[:,0],s=2.0**2)
		ax2.set_yscale("log")
		ax2.plot(h,0.5*h/T)
		plt.show()
	
def track_crossing(N, e, gamma, hmin,hmax, datafile="data/track_crossing",compute=False,read=True):	
	
	# This mode investigate the energy level 
	
	if(compute):
		pot=PotentialMP(e,gamma)
		idtmax=500
		
		T=np.array([])
		qE0=np.array([])
		qE1=np.array([])
		h=np.array([])

		h1=hmin
		h2=hmax

		#hmin
		grid=Grid(N,hmin)
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=idtmax)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		h=np.append(h,hmin)
		T=np.append(T,fo.getTunnelingPeriod())
		qE0=np.append(qE0,fo.getQEs(2)[0])
		qE1=np.append(qE1,fo.getQEs(2)[1])
		bool1=qE0[0]>qE1[0]
		
		grid=Grid(N,hmax)
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=idtmax)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		h=np.append(h,hmax)
		T=np.append(T,fo.getTunnelingPeriod())
		qE0=np.append(qE0,fo.getQEs(2)[0])
		qE1=np.append(qE1,fo.getQEs(2)[1])
		bool2=qE0[1]>qE1[1]
		
		dE=1.0
		i=0
		dEmax=10**(-15)
		print(dEmax)
		while dE>dEmax:
			print(dE/dEmax)
			hm=0.5*(h1+h2)

			grid=Grid(N,hm)
			wfcs=WaveFunction(grid)
			wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
			fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=idtmax)
			fo.diagonalize()
			fo.findTunellingStates(wfcs)
			
			h=np.append(h,hm)
			T=np.append(T,fo.getTunnelingPeriod())
			
			E0=fo.getQEs(2)[0]
			E1=fo.getQEs(2)[1]
			qE0=np.append(qE0,E0)
			qE1=np.append(qE1,E1)
			boolm=E0>E1
			
			#~ plt.plot(grid.x,np.real(fo.getEvec(0).x),c="blue",label="GS")
			#~ plt.plot(grid.x,np.real(fo.getEvec(1).x),c="red",label="1st ES")
			#~ plt.legend()
			#~ plt.show()
			
			if boolm==bool1:
				h1=hm
				bool1=boolm
			else:
				h2=hm
				bool2=boolm
				
			dE=abs(E1-E0)
				
		np.savez(datafile,"w", h=h, T=T,qE0=qE0,qE1=qE1)
	
	if(read):
		data=np.load(datafile) # pas zoomé
		fig, ax1 = plt.subplots()
		T=data['T']
		h=data['h']
		qE0=data['qE0']
		qE1=data['qE1']
		ax1.set_xlim(min(h),max(h))
		My=max(max(qE1),max(qE0))
		my=min(min(qE1),min(qE0))
		ax1.set_ylim(my,My)

		ax1.scatter(h,qE0,s=4.0**2,c="blue",label="0")
		ax1.scatter(h,qE1,s=4.0**2,c="red",label="1")
		ax1.legend()
		plt.show()

