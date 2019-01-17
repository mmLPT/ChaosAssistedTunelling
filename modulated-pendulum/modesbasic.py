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

def convert(s,nu):
	hbar=1.0545718 #e-34
	u=1.660538921 #e-27
	m=86.909180527*u 
	d=532.0 #e-9
	nuL=(np.pi*hbar)/(m*d**2)*10**(11)
	gamma=s*(nuL/nu)**2
	heff=2*(nuL/nu)
	return gamma, heff

def classical(pot,nperiod=100,ny0=20,wdir="classical/",compute=True):
	# If compute, is true, then it generate, save and plot SPS for a 
	# given MP potential. Otherwhise it simply plot SPS from wdir/
	cp=ClassicalContinueTimePropagator(pot)
	sb=StrobosopicPhaseSpaceMP(nperiod,ny0,cp,pot) #,pmax=np.pi)
	if compute:
		sb.save(wdir)
	sb.npz2plt(pot,wdir)
	
def propagate( grid, pot, iperiod, icheck,wdir,husimibool=False,wfbool=False,projfile="projs"):
	# Propagate a wavefunction over one period/one kick/on length
	# I periodically saves Husimi representation and wf in a .npz file
	
	husimi=Husimi(grid)
	fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=1000)

	projs=np.zeros(int(iperiod/icheck))
	time=np.zeros(int(iperiod/icheck))
	
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	wf0=WaveFunction(grid)
	wf0.setState("coherent",x0=-pot.x0, xratio=2.0)
	
	wf1=WaveFunction(grid)
	wf1.setState("coherent",x0=pot.x0, xratio=2.0)
	
	wf2=0.5*(wf1+wf0)
	
	fo.diagonalize()
	fo.findTunellingStates(wfcs)
	print(fo.getTunnelingPeriod())
	fo.getEvec(0).save("0")
	fo.getEvec(1).save("1")
	
	for i in range(0,iperiod):
		if i%icheck==0:
			print(str(i+1)+"/"+str(iperiod))
			time[int(i/icheck)]=i
			if husimibool:
				husimi.save(wf2,wdir+"husimi/"+strint(i/icheck))
			if wfbool:
				wf2.save(wdir+"wf/"+strint(i/icheck))
			projs[int(i/icheck)]=wf2.getx()
		fo.propagate(wf2)
	
	np.savez(wdir+projfile,"w", time=time, projs=projs)
	read.projs(wdir+projfile)
	
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
		
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i]=fo.getTunnelingPeriod()
		qE[i]=(fo.getQE(0),fo.getQE(1))
		
	np.savez(datafile,"w", h=h, T=T, qE=qE)
	
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
