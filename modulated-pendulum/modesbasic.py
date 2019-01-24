import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
import utils.plot.read as read

def getg(h):
			
	#nuX=25.0
	nuX=25.0
	nuL=8113.9
	nuperp=86
	
	a=5.0*10**(-9)
	d=532*10**(-9)
	Nat=10**5

	g=2*np.pi*h**2*a*Nat*nuperp/(2*nuL*d)
	print("g=",g)
	return 0.001

def getomegax(h):
	return (h*25.0)/(2*8113.9)

def convert2theory(s,nu):
	hbar=1.0545718 #e-34
	u=1.660538921 #e-27
	m=86.909180527*u 
	d=532.0 #e-9
	nuL=(np.pi*hbar)/(m*d**2)*10**(11)
	gamma=s*(nuL/nu)**2
	heff=2*(nuL/nu)
	return gamma, heff
	
def convert2exp(gamma,heff):
	hbar=1.0545718 #e-34
	u=1.660538921 #e-27
	m=86.909180527*u 
	d=532.0 #e-9
	nuL=(np.pi*hbar)/(m*d**2)*10**(11)
	nu=2*(nuL/heff)
	s=gamma/(nuL/nu)**2
	return s, nu

def classical(pot,nperiod=100,ny0=20,wdir="classical/",compute=True):
	# If compute, is true, then it generate, save and plot SPS for a 
	# given MP potential. Otherwhise it simply plot SPS from wdir/
	cp=ClassicalContinueTimePropagator(pot)
	sb=StrobosopicPhaseSpaceMP(nperiod,ny0,cp,pot) #,pmax=np.pi)
	if compute:
		sb.save(wdir)
	sb.npz2plt(pot,wdir)
	
def propagate( grid, pot, iperiod, icheck,wdir,projfile="projs"):
	# Propagate a wavefunction over one period/one kick/on length
	# I periodically saves Husimi representation and wf in a .npz file
	
	husimi=Husimi(grid)
	fo=CATFloquetOperator(grid,pot)
	n=int(iperiod/icheck)
	time=np.zeros(n)
	
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	for i in range(0,iperiod):
		if i%icheck==0:
			print(str(i+1)+"/"+str(iperiod))
			time[int(i/icheck)]=i
			#husimi.save(wf,wdir+"husimi/"+strint(i/icheck))
			#wf.save(wdir+"wf/"+strint(i/icheck))
		fo.propagate(wf)
	
	np.savez(wdir+projfile,"w", time=time, projs=projs)
	
def period_with_h(e=0.32, gamma=0.29, imax=520, N=128, datafile="split"):
	# Save tuneling period as a function of h
	h=np.linspace(0.15,0.41,imax)
	T=np.zeros(imax)
	qE=np.zeros((imax,2))

	for i in range(0,imax):
		print(str(i+1)+"/"+str(imax)+" - h="+str(h[i]))
		grid=Grid(N,h[i],2*np.pi)
		pot=PotentialMP(e,gamma)
			
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
		
		fo=CATFloquetOperator(grid,pot)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i]=fo.getTunnelingPeriod()
		qE[i]=(fo.getQE(0),fo.getQE(1))
		
	np.savez(datafile,"w", h=h, T=T, qE=qE)
	
def period_with_gamma(e, h,imax=250, N=64, compute=True, read=True, datafile="data/split-gamma-3"): #"data/split-gamma-2"
	if compute == True:
		# Save tuneling period as a function of h
		gamma=np.linspace(0.3083,0.3089,imax)
		print(convert2exp(gamma,h))
		T=np.zeros(imax)
		nstates=15
		qEs=np.zeros((imax,nstates))
		overlaps=np.zeros((imax,nstates))
		symX=np.zeros((imax,nstates))
		ind=np.zeros(nstates)
		
		
		grid=Grid(N,h)
		
		for i in range(0,imax):
			print(str(i+1)+"/"+str(imax)+" - g="+str(gamma[i]))
			
			pot=PotentialMP(e,gamma[i])
				
			wfcs=WaveFunction(grid)
			wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
			
			fo=CATFloquetOperator(grid,pot)
			fo.diagonalize()
			fo.computeOverlapsAndQEs(wfcs)
			T[i]=fo.getTunnelingPeriod()
			
			qEs[i],overlaps[i],symX[i],ind=fo.getQEsOverlapsIndex(nstates)
			
		np.savez(datafile,"w", e=e,h=h, gamma=gamma, T=T, qEs=qEs,overlaps=overlaps,symX=symX,imax=imax,nstates=nstates)
	
	if read == True:
		data=np.load(datafile+".npz")
		gamma=data['gamma']
		e=data['e']
		
		h=data['h']
		T=data['T']
		qEs=data['qEs']
		symX=data['symX']
		imax=data['imax']
		nstates=data['nstates']
		overlaps=data['overlaps']
		s,nu=convert2exp(gamma,h)
		ax=plt.gca()
		ax.set_yscale("log")
		
		ax.set_title(r"e=0.44, $\nu$=70.8")
		ax.set_xlabel(r"s")
		ax.set_ylabel(r"Période tunnel attendue")
		plt.scatter(s,2*T)
		plt.show()
		
		i1=np.argmax(overlaps[:,0])
		i2=np.argmax(overlaps[:,1])
		
		if symX[i1,0]==True:
			Nsym=overlaps[i1,0]
			Nasym=overlaps[i2,1]
		else:
			Nsym=overlaps[i2,1]
			Nasym=overlaps[i1,0]
		
		
		for i in range(0,nstates):
			cmapSym = plt.cm.get_cmap('Blues')
			cmapAsym = plt.cm.get_cmap('Reds')
			rgbaSym = cmapSym(overlaps[:,i]/Nsym)
			rgbaAsym = cmapAsym(overlaps[:,i]/Nasym)
			#print(rgba)
			for j in range(0,imax):
				if symX[j,i]==True:
					plt.scatter(gamma[j],qEs[j,i],c=rgbaSym,s=2.5**2)
				else:
					plt.scatter(gamma[j],qEs[j,i],c=rgbaAsym,s=2.5**2)
		plt.show()
	
def explore_epsilon_gamma(wdir="e_gamma/"):
	# Save tuneling period as a function of h for different values of
	# gamma and epsilon
	ngamma=7
	nepsilon=7
	N=64
	gamma=np.linspace(0.28,0.31,ngamma)
	epsilon=np.linspace(0.285,0.315,nepsilon)
	np.savez(wdir+"params","w", gamma=gamma, epsilon=epsilon,N=N)		
	for i in range(0,ngamma):
		for j in range(0,nepsilon):
			print("(",i,",",j,")")
			period_with_h(e=epsilon[j],gamma=gamma[i],N=N,datafile=wdir+"g"+str(i)+"e"+str(j),imax=260)
			
def true_full_sim():	
	gamma, h = modesbasic.convert(s=27.53, nu=70.8*10**3)
	e=0.44
	
	nuX=25.0
	nuL=8113.9
	#g=0.04
	g=0.00
	omegax=(h*nuX)/(2*nuL)
	
	N=256*8
	ncell=50
	iperiod=300
	icheck=1
	xmax=ncell*2*np.pi

	grid=Grid(N,h,xmax=xmax)
	pot=PotentialMPasym(e,gamma,0,omegax)
	
	wf=WaveFunction(grid)
	#wf.setState("load",datafile="true_sim/GS")
	wf.setState("coherent",xratio=1.0,x0=0.5*np.pi)
	#wf.shiftX(0.5*np.pi)
	
	fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=1000,g=g)
	#fo.propagateQuarterGP(wf,i0=0)
	
	
	for i in range(0,iperiod):
		if i%icheck==0:
			print(i)
			wf.save("true_sim/wf/"+strint(i/icheck))
			wf.savePNGx("true_sim/wfx-peak/"+strint(i/icheck),maxx0=1.0)
			#wf.savePNGp("true_sim/wfp/"+strint(i/icheck))
		fo.propagate(wf)
