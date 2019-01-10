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

def imaginary(compute=False,read=True):
	if(compute):
		N=1024*2
		gamma=0.3
		e=0.29
		h=0.3
		
		nuL=8113.9
		wL=2*np.pi*nuL
		d=532*10**(-9)
		
		nuperp=2.8*10**5
		
		a=5.*10**(-9)
		#g=4*np.pi*a*h**2*(2*np.pi)/d
		g=2*a*h*nuperp/nuL
		g=0.1
		
		print(g)

		xmax=20*2*np.pi
		grid=Grid(N,h,xmax=xmax)

		pot=PotentialMPasym(e,gamma,0,(grid.h*25.0)/(2*8113.9))
		
		qitp=QuantumImaginaryTimePropagator(grid,pot,T0=4*np.pi,idtmax=10000,g=g)
		
		wf=WaveFunction(grid)
		wf.setState("diracp")
		wf.setState("coherent",xratio=25.0)
		wf=qitp.getGroundState(wf)
		wf.save("gs0")
	if(read):
		read.wf("gs0")
	
def testImaginary():
	N=1024
	h=0.1
	g=20
	xmax=2*2*np.pi
	pot=PotentialTest()
	grid=Grid(N,h,xmax=xmax)
	qitp=QuantumImaginaryTimePropagator(grid,pot,T0=4*np.pi,idtmax=1000000,g=g)
	wf=WaveFunction(grid)
	husimi=Husimi(grid)
	wf.setState("diracp")
	wf.setState("coherent",xratio=5.0)
	wf=qitp.getGroundState(wf)
	wf.save("gs0")
	read.wf_special("gs0",N,xmax=xmax,g1=g)
