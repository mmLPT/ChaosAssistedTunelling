import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *

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
	return 0.04

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
	
def convert2exp(gamma,heff,x0=0.0):
	hbar=1.0545718 #e-34
	u=1.660538921 #e-27
	m=86.909180527*u 
	d=532.0 #e-9
	nuL=(np.pi*hbar)/(m*d**2)*10**(11)
	nu=2*(nuL/heff)
	s=gamma/(nuL/nu)**2
	x0exp=x0*180.0/np.pi
	return s, nu, x0exp

def classical(pot,nperiod=100,ny0=20,wdir="classical/",compute=True):
	# If compute, is true, then it generate, save and plot SPS for a 
	# given MP potential. Otherwhise it simply plot SPS from wdir/
	cp=ClassicalContinueTimePropagator(pot)
	sb=StrobosopicPhaseSpaceMP(nperiod,ny0,cp,pot) #,pmax=np.pi)
	if compute:
		sb.save(wdir)
	sb.npz2plt(pot,wdir)
	
def propagate( grid, pot, iperiod, icheck,compute=True,read=True,wdir="",datafile="free-prop"):
	# Propagate a wavefunction over one period/one kick/on length
	# I periodically saves Husimi representation and wf in a .npz file
	
	if(compute):
		fo=CATFloquetOperator(grid,pot)
		
		n=int(iperiod/icheck)
		xR=np.zeros(n)
		xL=np.zeros(n)
		time=np.zeros(n)
		
		wf=WaveFunction(grid)
		wf.setState("coherent",x0=-pot.x0,xratio=2.0)
		
		for i in range(0,iperiod):
			if i%icheck==0:
				print(str(i+1)+"/"+str(iperiod))
				xL[int(i/icheck)]=wf.getxL()
				xR[int(i/icheck)]=wf.getxR()
				time[int(i/icheck)]=i*2
			fo.propagate(wf)
		
		np.savez(wdir+datafile,"w", time=time, xL=xL,xR=xR,n=n,paramspot=pot.getParams(),h=grid.h)
		
		if(read):
			data=np.load(wdir+datafile+".npz")
			e=data['paramspot'][0]
			gamma=data['paramspot'][1]
			x0=data['paramspot'][2]
			h=data['h']
			s,nu,x0exp=modesbasic.convert2exp(gamma,h,x0)
			
			
			time=data['time']
			n=data['n']
			xR=data['xR']
			xL=data['xL']

			plt.plot(time,xL, c="red")
			plt.plot(time,xR, c="blue")
			
			ax=plt.gca()
			ax.set_xlabel(r"Périodes")
			ax.set_ylabel(r"$x gauche et x droite$")
			ax.set_title(r"$s={:.2f} \quad \nu={:.2f}\ kHz \quad \varepsilon={:.2f}  \quad x_0={:.0f}^\circ$".format(s,nu*10**(-3),e,x0exp))
			ax.set_xlim(0,max(time))
			
			plt.show()
	
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
	
def period_with_gamma(e, h,imax=500, N=64, compute=False, read=True, datafile="data/split-gamma-3"): #"data/split-gamma-2"
	if compute == True:
		# Save tuneling period as a function of h
		gamma=np.linspace(0.25,0.35,imax)
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
		print(e,h)
		T=data['T']
		qEs=data['qEs']
		symX=data['symX']
		imax=data['imax']
		nstates=data['nstates']
		overlaps=data['overlaps']
		s,nu,x0=convert2exp(gamma,h,0)
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
					plt.scatter(gamma[j],qEs[j,i],c=rgbaSym[j],s=2.5**2)
				else:
					plt.scatter(gamma[j],qEs[j,i],c=rgbaAsym[j],s=2.5**2)
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
		
		
def check_projection(grid, pot,iperiod=100,compute=True,read=True,wdir="CAT/"):
	# Propagate a wavefunction over one period/one kick/on length
	# I periodically saves Husimi representation and wf in a .npz file
	
	#~ double=False
	#~ cp=ClassicalContinueTimePropagator(pot)
	#~ sb=StrobosopicPhaseSpaceMP(1000,10,cp,pot) #,pmax=np.pi)
	#~ sb.save(wdir+"class/",double=double)
	#~ sb.npz2plt(pot,wdir+"class/",double=double)
	
	
	x0=0.5*np.pi

	time=np.zeros(iperiod)
	projs=np.zeros((3,iperiod))
	x=np.zeros((2,iperiod))
	
	fo=CATFloquetOperator(grid,pot,beta=-0.5*grid.h/6.0)
	
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	fo.diagonalize()
	fo.computeOverlapsAndQEs(wf)
	wf1,wf2,wf3=fo.get3Evec()
	
	sym1=WaveFunction(grid)
	sym1.setState("coherent",x0=x0,xratio=2.0)
	sym2=WaveFunction(grid)
	sym2.setState("coherent",x0=-x0,xratio=2.0)
	
	sym=sym1+sym2
	sym.normalizeX()
	sym.x2p()
			
	
	wf4=(sym%wf2)*wf2+(sym%wf3)*wf3
	wf5=-(sym%wf3)*wf2+(sym%wf2)*wf3
	plt.plot(grid.x,abs(wf4.x)**2)
	plt.plot(grid.x,abs(wf5.x)**2)
	plt.show()
	
	
	#~ wf4=(sym%wf1)*wf2+(sym%wf3)*wf3
	#~ wf5=-(sym%wf3)*wf2+(sym%wf1)*wf3
	#~ plt.plot(grid.x,abs(wf4.x)**2)
	#~ plt.plot(grid.x,abs(wf5.x)**2)
	#~ plt.show()
	
	#~ husimi=Husimi(grid)
	#~ husimi.save(wf4,wdir+"husimi/1")
	#~ husimi.npz2png(wdir+"husimi/1")
	#~ husimi.save(wf5,wdir+"husimi/2")
	#~ husimi.npz2png(wdir+"husimi/2")
	
	plt.plot(grid.x,np.real(wf1.x),c="red")
	plt.plot(grid.x,np.real(wf2.x),c="blue")
	plt.plot(grid.x,np.real(wf3.x),c="green")
	plt.show()
	
	for i in range(0,iperiod):
		time[i]=2*i
		fo.propagate(wf)
		projs[:,i]=wf//wf1,wf//wf4,wf//wf5
		x[:,i]=wf.getxL(),wf.getxR()
	
	#~ np.savez(wdir+"projs","w", projs=projs,time=time)
	
	plt.plot(time,projs[1,:])
	plt.plot(time,projs[2,:])
	plt.show()
	
	plt.plot(time,x[0,:])
	plt.plot(time,x[1,:])
	plt.show()
