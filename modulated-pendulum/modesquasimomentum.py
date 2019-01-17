import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
#from utils.mathtools.periodicfunctions import *
import utils.plot.read as readw
import modesbasic

def imaginary(gamma, e,h, datafile="true_sim/GS",compute=True,read=True):
	if(compute):
		N=256*8

		
		#nuX=25.0
		nuX=25.0
		nuL=8113.9
		nuperp=86
		
		a=5.0*10**(-9)
		d=532*10**(-9)
		Nat=10**5

		g=2*np.pi*h**2*a*Nat*nuperp/(2*nuL*d)
		print("g=",g)
		g=0.04
		
		#gamma=0.0
		
		ncell=50
		xmax=ncell*2*np.pi
		grid=Grid(N,h,xmax=xmax)
		
		omegax=(grid.h*nuX)/(2*nuL)
		pot=PotentialMPasym(e,gamma,0,omegax)
		
		qitp=QuantumImaginaryTimePropagator(grid,pot,T0=4*np.pi,idtmax=100000,g=g)
		
		wf=WaveFunction(grid)
		#wf.setState("diracp")
		wf.setState("coherent",xratio=ncell)
		#wf.setState("random")
		wf=qitp.getGroundState(wf)
		wf.save(datafile)
		np.savez(datafile+"-dat","w", omegax=omegax, h=h, gamma=gamma, e=e,g=g,weight=grid.intweight)
		
	if(read):
		data=np.load(datafile+".npz")
		x=data["x"]
		p=data["x"]
		psix=data["psix"]

		data2=np.load(datafile+"-dat.npz")
		heff=data2["h"]
		omegax=data2["omegax"]
		weight=data2["weight"]
		x0=0.512/(2*np.pi) #Back to true lenght (in micrometer)
		
		# Harmonic oscillator with no interactions
		sigmax2=heff/(2*omegax)
		phi_th=(2*np.pi*sigmax2)**(-0.25)*np.exp(-x**2/(4*sigmax2))
		plt.plot(x*x0,abs(phi_th)**2)
		print(abs(sum(np.conj(phi_th)*phi_th))*weight)
		
		ax=plt.gca()
		ax.set_xlabel(r"$x$ ($\mu$m)")
		ax.set_ylabel(r"$|\psi(x)|^2$")
		ax.set_yticks([])		
		
		plt.plot(x*x0,abs(psix)**2)
		print(abs(sum(np.conj(psix)*psix))*weight)
		plt.show()
		
	if(not(read) and not(compute)):
		data1=np.load("data/GS-sans-interactions-1.npz")
		x=data1["x"]
		psix1=data1["psix"]
		data2=np.load("data/GS-avec-interactions-1.npz")
		psix2=data2["psix"]
		
		x0=0.512/(2*np.pi) #Back to true lenght (in micrometer)
		
		ax=plt.gca()
		ax.set_xlabel(r"$x$ ($\mu$m)")
		ax.set_ylabel(r"$|\psi(x)|^2$ ")
		ax.set_yticks([])
		
		plt.plot(x*x0,abs(psix1)**2)
		plt.plot(x*x0,abs(psix2)**2)
		plt.show()
		
def free_prop_averaged(grid, pot,x0,compute=False,read=True,datafile="data/qm-exp1"):
	if(compute):
		iperiod=150
		icheck=1
		n=int(iperiod/icheck)
		
		ibetamax=100
		
		xpos=np.zeros((ibetamax,n))
		time=np.zeros(n)

		
		for j in range(0,ibetamax):
			beta=np.random.normal(0.0, 0.02*grid.h)
			if ibetamax==1:
				beta=0.0
			print(beta)
			fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=1000,beta=beta)
			
			wf=WaveFunction(grid)
			wf.setState("coherent",x0=x0,xratio=2.0)
			
			for i in range(0,iperiod):
				if i%icheck==0:
					print(j,"-",i)
					xpos[j][int(i/icheck)]=wf.getx()
					#print(wf.getx())
					#plt.plot(abs(wf.x)**2)
					plt.show()
					print(xpos[j][int(i/icheck)])
					time[int(i/icheck)]=i*2
				fo.propagate(wf)
			
		np.savez(datafile,"w", time=time, xpos=xpos,ibetamax=ibetamax,n=n)
	if(read):
		data=np.load(datafile+".npz")
		xpos=data['xpos']
		time=data['time']
		ibetamax=data['ibetamax']
		n=data['n']
		print(xpos[0,:])
		##np.average(time,np.mean(xpos,axis=0))
		meanpos=np.mean(xpos,axis=0)
		plt.plot(time,meanpos)
		
		ax=plt.gca()
		ax.set_xlabel(r"Périodes")
		ax.set_ylabel(r"$\langle \psi | \hat{X} | \psi \rangle$")
		ax.set_title(r"$s=27.53 \ \nu=70.8 \ kHz \ \varepsilon=0.44 \ x_0=0.5 \pi$")
		
		plt.show()

		f= open("simu-premiere-oscillation.txt","w+")
		f.write("  t \t     x\n =============== \n")
		for i in range(0,meanpos.size):
			f.write("{0:4.0f}\t {1:+7.5f}\n".format(time[i],meanpos[i]))
			
		f.close()
		
		
def true_sim(compute=True,read=True,datafile="data/qm-exp1"):
	if(compute):
		iperiod=150
		icheck=1
		n=int(iperiod/icheck)
		
		
		pt=np.zeros(n)
		time=np.zeros(n)

		for i in range(0,iperiod):
			if i%icheck==0:
				print(i)
				wf.save("true_sim/"+strint(i/icheck))
				time[int(i/icheck)]=i*2
			fo.propagate(wf)
			
		np.savez(datafile,"w", time=time, xpos=xpos,ibetamax=ibetamax,n=n)

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
