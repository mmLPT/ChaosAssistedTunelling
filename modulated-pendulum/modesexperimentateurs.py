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

def prediction_periode(s,mu,epsilon,ds=0.02,dmu=0.02,n=2):
	
	gamma0=
	h0=
	
	N=128
	s=np.linspace(h*(1-dheff),h*(1+dheff),n)
	nuL=np.linspace(h*(1-dheff),h*(1+dheff),n)
	
	
	for i in range(0,imax):
		grid=Grid(N,h[i],2*np.pi)
		pot=PotentialMP(e,gamma)
			grid=Grid(N,h[i],2*np.pi)
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
		
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i]=fo.getTunnelingPeriod()
		qE[i]=(fo.getQE(0),fo.getQE(1))
		
	np.savez(datafile,"w", h=h, T=T, qE=qE)
