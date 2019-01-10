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

def classical(pot,nperiod=100,ny0=20,wdir="classical/"):
	# Classical stroboscopic phase
	cp=ClassicalContinueTimePropagator(pot)
	sb=StrobosopicPhaseSpaceMP(nperiod,ny0,cp,pot) #,pmax=np.pi)
	sb.save(wdir)
	sb.npz2plt(pot,wdir)
	print(pot.R1())
	
def propagate( grid, pot, iperiod, icheck,wdir,husimibool=False,wfbool=False,projfile="projs"):
	# Propagate a wavefunction over one period/one kick/on length
	# I periodically saves Husimi representation and wf in a .npz file
	husimi=Husimi(grid)

	fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=1000)

	
	projs=np.zeros(int(iperiod/icheck))
	time=np.zeros(int(iperiod/icheck))
	
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=-pot.x0, xratio=2.0)
	
	for i in range(0,iperiod):
		if i%icheck==0:
			print(str(i+1)+"/"+str(iperiod))
			time[int(i/icheck)]=i
			if husimibool:
				husimi.save(wf,wdir+"husimi/"+strint(i/icheck))
			if wfbool:
				wf.save(wdir+"wf/"+strint(i/icheck))
			projs[int(i/icheck)]=wfcs//wf
		fo.propagate(wf)
	
	np.savez(wdir+projfile,"w", time=time, projs=projs)
	read.projs(wdir+projfile)
	
def period_with_h(e=0.32, gamma=0.29, imax=520, N=128, datafile="split"):
	# Save tuneling period as a function of h
	# For symetric potential
	#h=np.linspace(1/10.0,1/7.5,imax)
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
	
def read_period_with_h(datafile="split"):
	data=np.load(datafile+".npz")
	h=data['h']
	T=data['T']
	plt.xlim(min(h),max(h))
	plt.yscale('log')
	plt.plot(h,T)
	plt.show()
	
def explore_epsilon_gamma(wdir="e_gamma/"):
	ngamma=7
	nepsilon=7
	N=128*2
	gamma=np.linspace(0.28,0.31,ngamma)
	epsilon=np.linspace(0.285,0.315,nepsilon)
	for i in range(0,ngamma):
		for j in range(0,nepsilon):
			print("(",i,",",j,")")
			period_with_h(e=epsilon[i],gamma=gamma[i],N=N,datafile=wdir+"g"+str(i)+"e"+str(j))
	np.savez(wdir+"params","w", gamma=gamma, epsilon=epsilon,N=N)		

		
def mode2(grid,e,gamma,wdir="asym/",datafile="data"):
	# potentiel asymetrique
	# on fait varier x1, on regarde la periode
	
	def e1(N,pot,fo,evec0,i0):
		e1=0.0
		for i in range(0,N):
			if not(i==i0):
				eveci=fo.getEvec(i)[1]
				e1+=np.abs(pot.braketVxasym(eveci,evec0))**2/fo.diffqE1qE2(i0,i)
		return e1
	# Regarde comment evolue la periode
	npoints=11
	x1=np.linspace(-30*2*np.pi,30*2*np.pi,npoints)
	
	T=np.zeros(npoints)
	Tth=np.zeros(npoints)
	qE=np.zeros((npoints,2))
	qEth=np.zeros((npoints,2))	
	
	husimi=Husimi(grid)
	pot=PotentialMPasym(e,gamma,0,0.0) #(grid.h*25.0)/(2*8113.9))
	wfcsp=WaveFunction(grid)
	wfcsp.setState("coherent",x0=pot.x0,xratio=2.0)
	
	wfcsm=WaveFunction(grid)
	wfcsm.setState("coherent",x0=-pot.x0,xratio=2.0)
	
	
	
	# Initial state
	fo0=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
	fo0.diagonalize()
	fo0.findTunellingStates(wfcsp)
	qE0=np.zeros(2)
	T0,qE0=fo0.getSplitting()

	for i in range(0,npoints):
		
		pot=PotentialMPasym(e,gamma,x1[i],(grid.h*25.0)/(2*8113.9))
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		
		if x1[i]>0.0:
			fo.findTunellingStates(wfcsp)
		else:
			fo.findTunellingStates(wfcsm)
			
		T[i],qE[i]=fo.getSplitting()
		qEth[i][0]=qE0[0] +np.real(pot.braketVxasym(wf0,wf0))+np.abs(np.real(pot.braketVxasym(wf1,wf0)))**2/(qE0[0]-qE0[1])
		qEth[i][1]=qE0[1] +np.real(pot.braketVxasym(wf1,wf1)) +np.abs(np.real(pot.braketVxasym(wf1,wf0)))**2/(qE0[1]-qE0[0])
		Tth[i]=0.5*grid.h/(qEth[i][1]-qEth[i][0])

		#print("x=",x1[i],"T=",T[i],"T_th=",Tth[i],"qE1=",qE[i][0],"qE1_th=",qEth[i][0],"qE2=",qE[i][1],"qE2_th=",qEth[i][1])
	np.savez(wdir+datafile,"w", x1=x1, T=T, Tth=Tth,qE=qE,qEth=qEth,qE0=qE0)

def check_T_with_confinment(imax=11, e=0.32, gamma=0.29,N=64, xasym=15*2*np.pi, datafile="split"):
	#
	
	h=np.linspace(0.2090024,0.209008,imax)
	
	T0=np.zeros(imax)
	Tasym=np.zeros(imax)
	isLowerSymetric=np.zeros(imax, dtype=bool)

	for i in range(0,imax):
		
		# Initialization
		grid=Grid(N,h[i],2*np.pi)
		pot0=PotentialMP(e,gamma)
		potasym=PotentialMPasym(e,gamma,xasym,(grid.h*25.0)/(2*8113.9))
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot0.x0,xratio=2.0)
		
		# Harmonic potential at x=0
		fo0=CATFloquetOperator(grid,pot0,T0=4*np.pi,idtmax=500)
		fo0.diagonalize()
		fo0.findTunellingStates(wfcs)
		T0[i]=fo0.getTunnelingPeriod()
		qE0[i]=fo0.getQE(0),fo0.getQE(1)
		isLowerSymetric[i]=fo0.getEvec(0).isSymetricInX()

		# Harmonic potential at x=xshift
		foasym=CATFloquetOperator(grid,potasym,T0=4*np.pi,idtmax=500)
		foasym.diagonalize()
		foasym.findTunellingStates(wfcs)
		Tasym[i],qEasym[i]=foasym.getSplitting()
		dqEasym=foasym.getSplittingE()
		print(str(i+1),"/",imax,"h=",h[i],"Tasym=",Tasym[i],"T0=",T0[i])
		
	np.savez(datafile,"w", h=h, T0=T0, Tasym=Tasym, isLowerSymetric=isLowerSymetric)
	
def symetry_of_gs_with_h(N, e, gamma, datafile="split"):
	imax=100
	h=np.linspace(0.1,0.40,imax)
	T=np.zeros(imax)
	Tasym=np.zeros(imax)
	qE=np.zeros((imax,2))
	isLowerSymetric=np.zeros(imax, dtype=bool)
	isPeriodGrowing=np.zeros(imax, dtype=bool)

	for i in range(0,imax):
		print(str(i+1)+"/"+str(imax)+" - h="+str(h[i]))
		grid=Grid(N,h[i],2*np.pi)
		pot=PotentialMP(e,gamma)
		potasym=PotentialMPasym(e,gamma,15*2*np.pi,(grid.h*25.0)/(2*8113.9))
			
		wfcs=WaveFunction(grid)
		wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
		
		fo=CATFloquetOperator(grid,pot,T0=4*np.pi,idtmax=500)
		fo.diagonalize()
		fo.findTunellingStates(wfcs)
		T[i]=fo.getTunnelingPeriod()
		qE[i]=fo.getQE(0),fo.getQE(1)
		isLowerSymetric[i]=fo.getEvec(0).isSymetricInX()
		
		foasym=CATFloquetOperator(grid,potasym,T0=4*np.pi,idtmax=500)
		foasym.diagonalize()
		foasym.findTunellingStates(wfcs)
		Tasym[i]=foasym.getTunnelingPeriod()
		isPeriodGrowing[i]=Tasym[i]>T[i]
		print(isPeriodGrowing[i],isLowerSymetric[i])
	
	np.savez(datafile,"w", h=h, T=T,qE=qE, isLowerSymetric=isLowerSymetric,isPeriodGrowing=isPeriodGrowing)


"""
nuL=8113.9
nux=25.0

N=1024#128+32+16+4+2+1+1
hold=0.3
h=(2*nuL/nux)
gamma=0.29*(h/hold)**2
e=0.35
omega=.1


xmax=20*2*np.pi
grid=Grid(N,h,xmax=xmax)


a=5.0#e-9
m=1.45 #e-25
hreal=6.62607015 #/(2*np.pi)  #e-34

g=2*grid.h*(hreal*a/m*1.0e-18)/(2*np.pi*nux)
print(g, h, gamma, (omega*xmax)**2)
#print(h*nux**2/(2.0*nuL))

pot=PotentialMPasym(e,gamma,0,omega)
qitp=QuantumImaginaryTimePropagator(grid,pot,T0=4*np.pi,idtmax=10000,g=g)
wf=WaveFunction(grid)
husimi=Husimi(grid)
wf.setState("diracp")
#wf.setState("coherent",xratio=25.0)
wf=qitp.getGroundState(wf)
wf.save("gs0")
read.wf("gs0")
"""
"""
# Pour OH + non linearite
N=1024#128+32+16+4+2+1+1
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

"""
