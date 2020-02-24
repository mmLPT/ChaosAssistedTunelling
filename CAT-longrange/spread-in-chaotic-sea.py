import sys
sys.path.insert(0, '..') 
sys.path.insert(0, '/users/martinez/Documents/Cloud CNRS/Plot') 
import latex 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.systems.kickedrotor import *
from matplotlib import gridspec

from scipy.optimize import curve_fit

from hamTB import *

mode=sys.argv[1]

# ~ datafile="tempdata/hm2d112-Nc351"
# ~ datafile="tempdata/hm2d112-Nc117"
# ~ datafile="tempdata/hm2d112-Nc39"
# ~ datafile="tempdata/hm2d112-Nc13"

datafile="tempdata/hm2d112-bloch-Nc1079-F0d000025"
datafile="tempdata/hm2d112-bloch-Nc179-F0d00005"
datafile="tempdata/hm2d112-bloch-Nc1079-F0d00005"
datafile="tempdata/hm2d112-bloch-Nc1079-F0d000001"

if mode=="compute":
	h=1/2.112
	e=0.50
	gamma=0.15
	tmax=20000
	Npcell=32
	Ncell=1079
	F=0.000001
	N=Ncell*Npcell
	itmax=int(tmax/2) 
	
	print("Nh=",N*h/Ncell)

	# Creation of objects
	pot=PotentialMP(e,gamma,Fb=-F)
	grid=Grid(N,h,xmax=Ncell*2*np.pi)
	fo=CATFloquetOperator(grid,pot)
	time=np.linspace(0.0,2.0*itmax,num=itmax,endpoint=False)
	
	# Index
	ind=np.arange(Ncell)
	indcell=np.arange(-int((Ncell-1)/2),int((Ncell-1)/2+1))
	
	### Construction état initial ######################################
	
	# Paramètres
	n0=0 # position initiale
	ctype="gaussienne"
	# ~ ctype="mono"
	dn=200

	# 1. chargement structure de bande
	dataTBreg=sys.argv[2]
	tb=TightBinding(dataTBreg,Ncell)
	
	# 2. construction de l'état
	wf=WaveFunction(grid)
	if ctype=="gaussienne":
		for icell in ind:
			wfEBK=WaveFunction(grid)	
			wfEBK.x=np.abs(tb.wf0.x)
			wfEBK.x2p()
			
			wf.x=wf.x+np.roll(wfEBK.x,-indcell[icell]*Npcell)*np.exp(-(indcell[icell]-n0)**2/dn)
	elif ctype=="mono":
		wf.x=np.abs(tb.wf0.x)
	wf.normalizeX()
	wf.x2p()
	
	# 3. affichage
	ax=plt.subplot(2,1,1)
	ax.set_xlim(-30,30)
	ax.plot(grid.x/(2*np.pi),np.real(wf.x),c='blue')	
	ax.plot(grid.x/(2*np.pi),np.imag(wf.x),c='red')	
	ax=plt.subplot(2,1,2)
	ax.plot(grid.p,np.real(wf.p),c='blue')	
	ax.plot(grid.p,np.imag(wf.p),c='red')	
	plt.show()
	
	### Simulation #####################################################

	# 1. observables
	prob=np.zeros((itmax,Ncell))
	xstd=np.zeros(itmax)
	xm=np.zeros(itmax)
	
	# 2. construction état pour projeter 
	wficell=[]
	for icell in ind:
		wficell.insert(icell,WaveFunction(grid))
		wficell[icell].x=np.roll(np.abs(tb.wf0.x),-indcell[icell]*Npcell)

	# 3. dynamique	
	for it in range(0,itmax):
		print(it,"/",itmax)
		for icell in ind:
			prob[it,icell]=wf//wficell[icell]

		xstd[it]=np.sqrt(np.sum(indcell**2*prob[it])/np.sum(prob[it]))
		xm[it]=np.sum(indcell*prob[it])/np.sum(prob[it])
		fo.propagate(wf)
	
	### Sauvegarde #####################################################	
	np.savez(datafile,F=F,h=h,e=e,gamma=gamma,prob=prob,time=time,xstd=xstd,xm=xm,Ncell=Ncell)	



# ~ datafile="tempdata/spread-in-chaotic-sea-hmd3d781"

# ~ datafile="tempdata/hm2d112-Nc27"

# ~ datafile="tempdata/hm2d112"

datafile="tempdata/hm2d112-Nc351"
	
if mode=="plotTB":
	data=np.load(datafile+".npz")
	time=data['time']
	prob=data['prob']
	xstd=data['xstd']
	xm=data['xm']
	F=data['F']
	h=data['h']
	Ncell=int(data['Ncell'])
	data.close()
	
	# ~ F=0.00005
	# ~ h=1/2.112
	
	# ~ F=0
	
	
	
	
	Ncell=prob.shape[1]
	itmax=time.size
	print(itmax)
	
	# ~ print(Ncell)
	
	dataTB=sys.argv[2]

	tb=TightBinding(dataTB,Ncell,-F)
	U=tb.U
	Ncell=tb.Ncell
	n0TB=int(0.5*(Ncell-1))
	tb.wf[n0TB]=1
	
	# ~ tb.wf=np.exp(-(np.arange(Ncell)-n0TB)**2/250)
	# ~ tb.wf=tb.wf/np.sqrt(np.sum(np.abs(tb.wf)**2))
	
	print(np.sum(prob,axis=1))
	
	for i in range(0,itmax):
		prob[i,:]=prob[i,:]/np.sum(prob[i,:])
		
	print(np.sum(prob,axis=1))
	
	
	# Observables
	probTB=np.zeros((itmax,Ncell))
	xstdTB=np.zeros(itmax)
	xmeanTB=np.zeros(itmax)
	
	for it in range(0,itmax):
		probTB[it]=np.abs(tb.wf)**2
		xstdTB[it]=tb.xstd()
		xmeanTB[it]=tb.xmean()
		# ~ print(xmeanTB[it])
		tb.propagate()
	
	
	# FILM  A PARTIR DU TB
	
	# ~ ax=plt.gca()
	# ~ dn=50
	# ~ n=np.arange(dn)
	# ~ for i in range(0,2500,10):
		# ~ ax.set_ylim(0,0.15)
		# ~ ax.set_xlim(0,dn)
		# ~ ax.scatter(n,probTB[i,n0TB:n0TB+dn],c="blue",zorder=1,label="TB",s=3**2)
		# ~ prob0=np.zeros(1500)
		# ~ x=np.linspace(-0,dn,1500)
		# ~ for j in n:
			# ~ prob0+=prob[i,n0TB+j]*np.exp(-(x-j)**2/0.01)
			
		# ~ ax.plot(x,prob0,c="red",zorder=0,label="Ncell")
		#ax.legend()
		# ~ plt.savefig("tempdata/movie/"+strint(i/10)+".png", bbox_inches='tight',dpi=250)
		# ~ ax.clear()

	# ~ probIni=np.sum(prob[:,indIni],axis=1)
	# ~ probElse=np.sum(prob[:,indElse],axis=1)
	
	
	tmax=np.max(time)

	# ~ fig = plt.gcf()
	# ~ fig.suptitle(r"$1/h={:.3f} \quad N_p={:d}$".format(float(1/h),int(Ncell)))

	
	####################################################################
	
	# ~ ax=plt.subplot(2,2,1)
	# ~ ax.set_xlabel("time")
	# ~ ax.set_ylabel(r"$|<n_0|\psi>|^2$")
	# ~ ax.set_xlim(0,tmax)
	# ~ ax.scatter(time,prob[:,n0TB],c="blue",s=2**2,label="Exact")
	# ~ ax.plot(time,probTB[:,n0TB],c="red",label="TB")
	# ~ ax.legend()
	
	# ~ ax=plt.subplot(2,2,3)	
	# ~ ax.set_xlabel("n")
	# ~ ax.set_ylabel(r"$|V_n|$")
	# ~ ax.set_yscale('log')
	# ~ ax.scatter(n,np.abs(Vn),label="TB")
	# ~ ax.scatter(n,func(n,*popt),label="fit 1/n")
	# ~ ax.legend()
	
	
	# ~ ax=plt.subplot(2,2,2)	
	# ~ ax.set_xlabel("time")
	# ~ ax.set_ylabel(r"$\sqrt{<(x-<x>)^2>}$")
	# ~ ax.set_xlim(0,tmax)
	# ~ ax.plot(time,xstd,c="blue",label="Exact")
	# ~ ax.plot(time,xstdTB,c="red",label="TB")
	# ~ ax.legend()

	
	# ~ diff0=np.median(np.abs(prob-probTB)/prob,axis=1)
	# ~ diff0=np.median(np.abs(probTB)/prob,axis=1)
	# ~ diff1=np.median(np.abs(prob-probTB)/np.mean(prob,axis=0),axis=1)

	# ~ ax=plt.subplot(2,2,4)	
	# ~ ax.set_xlabel("time")
	# ~ ax.set_ylabel("Médiane erreur (sur les sites)")
	# ~ ax.set_xlim(0,tmax)
	# ~ ax.set_ylim(0,0.5)
	# ~ ax.plot(time,diff0,c="blue",label="Relative")
	# ~ ax.plot(time,diff1,c="red",label="Absolu normalisée")
	# ~ ax.grid()
	# ~ ax.legend()
	
	# ~ plt.show()
	
	####################################################################
	
	fig= latex.fig(columnwidth=452.96,wf=1,hf=0.25)
	gs = gridspec.GridSpec(2,2, hspace = 0,wspace = 0.2)
	
	x = time
	y = np.linspace(-Ncell/2, Ncell/2, Ncell)
	X, Y = np.meshgrid(x, y)
	levels = np.linspace(0,1,10,endpoint=True)

	ax=plt.subplot(gs[0,0])
	
	ax.text(0.02,0.93, r"$\bold{(a)}$", transform=ax.transAxes, fontsize=6,va='top',ha='left')
	
	ax.set_ylim(0,150)
	ax.set_yticks([75,150])
	
	ax.set_ylabel(r"$\|\langle n|\Psi(t) \rangle\|^2$")
	ax.yaxis.set_label_coords(-0.15, 0)
	ax.set_xticks([])
	

	Z=np.swapaxes(np.abs(prob)/np.max(np.abs(prob),axis=0),0,1)
	im=ax.contourf(X, Y, Z, levels=levels, cmap='gnuplot2_r',extent=(0,0,np.max(time),100))
	
	ax=plt.subplot(gs[1,0])
	
	ax.text(0.02,0.07, r"$\bold{(b)}$", transform=ax.transAxes, fontsize=6,va='bottom',ha='left')
	
	ax.set_xlabel(r"Time $t$")
	
	ax.set_ylim(-150,0)
	ax.set_yticks([0,-75,-150])
	
	Z=np.swapaxes(np.abs(probTB)/np.max(np.abs(probTB),axis=0),0,1)
	im=ax.contourf(X, Y, Z, levels=levels, cmap='gnuplot2_r',extent=(0,0,np.max(time),100))
	
	
	ax=plt.subplot(gs[0:2,1])
	
	ax.text(0.98,0.95, r"$\bold{(c)}$", transform=ax.transAxes, fontsize=6,va='top',ha='right')
	
	dataTB=sys.argv[2]
	tb=TightBinding(dataTB)
	
	Vn=tb.Vn
	Vn=np.delete(Vn,0)
	
	n=np.arange(Vn.size,dtype=float)+1
	
	ax.set_xlabel(r"$n$")
	ax.set_ylabel(r"$|V_n|$")
	ax.set_yscale('log')
	ax.scatter(n,np.abs(Vn),label=r"$|Vn|^2$",c="blue",s=2**2,zorder=15)
	ax.set_xlim(0,150)
	ax.set_ylim(10**(-6),10**(-3))
	ax.set_yticks([10**(-6),10**(-3)])
	# ~ ax.grid(which="minor")
	
	ax.yaxis.set_label_coords(-0.05, 0.5)
	
	
	

	# ~ ax=plt.subplot(4,1,2)
	
	# ~ Z=np.swapaxes(0.5*np.abs(probTB)/prob,0,1)
	# ~ im=ax.contourf(X, Y, Z, levels=levels, cmap='gnuplot2')
	# ~ ax.set_xlabel("time")
	# ~ ax.set_ylabel("|TB-exact|/exact")
	
	# ~ ax=plt.subplot(4,1,3)
	
	# ~ Z=np.swapaxes(np.abs(prob-probTB)/np.mean(prob),0,1)
	# ~ im=ax.contourf(X, Y, Z, levels=levels, cmap='gnuplot2')
	# ~ ax.set_xlabel("time")
	# ~ ax.set_ylabel("|TB-exact| normalisé")

	# ~ ax=plt.subplot(4,1,4)
	# ~ ax.set_xlabel("time")
	# ~ ax.set_ylabel(r"$\sqrt{<(x-<x>)^2>}$")
	# ~ ax.set_xlim(0,tmax)
	# ~ ax.plot(time,xstd,c="blue",label="Exact")
	# ~ ax.plot(time,xstdTB,c="red",label="TB")
	# ~ ax.legend()
	
	# ~ fig.subplots_adjust(right=0.8)
	# ~ cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	# ~ fig.colorbar(im, cax=cbar_ax)
	
	latex.save("tb","pdf")
	# ~ plt.show()
	
	####################################################################

	
# ~ datafile="tempdata/bloch2"#F=0.0001/reg
# ~ datafile="tempdata/bloch1" #F=0.0001/chaotic
# ~ datafile="tempdata/bloch3" #F=0.00005/chaotic
# ~ datafile="tempdata/hm2d112-bloch-Nc1079-F0d000025"	
# ~ datafile="tempdata/hm2d112-bloch-Nc179-F0d00005"

# ~ datafile="tempdata/hm2d112-bloch-Nc1079-F0d000025"
# ~ datafile="tempdata/hm2d112-bloch-Nc179-F0d00005"
# ~ datafile="tempdata/hm2d112-bloch-Nc1079-F0d00005"
	
if mode=="plotbloch":
	data=np.load(datafile+".npz")
	time=data['time']
	prob=data['prob']
	xstd=data['xstd']
	xm=data['xm']
	F=data['F']
	h=data['h']
	Ncell=int(data['Ncell'])
	data.close()
	
	print(F)
	
	# ~ F=0.00005
	# ~ F=0.0001
	# ~ h=1/2.112
	Ncell=prob.shape[1]
	itmax=time.size
	
	# ~ for i in range(0,itmax):
		# ~ prob[i,:]=prob[i,:]/np.sum(prob[i,:])
		
	
	# ~ ax=plt.gca()
	# ~ ax.plot(prob[1200])
	# ~ plt.show()
	
	dataTB=sys.argv[2]
	tb=TightBinding(dataTB)
	####################################################################
	
	x = time
	y = np.linspace(-Ncell/2, Ncell/2, Ncell)
	X, Y = np.meshgrid(x, y)
	levels = np.linspace(0,1,10,endpoint=True)

	ax=plt.subplot(2,1,1)
	
	ax.set_ylabel("exact")

	Z=np.swapaxes(np.abs(prob)/np.max(np.abs(prob)),0,1)
	ax.contourf(X, Y, Z, levels=levels, cmap='gnuplot2')
	
	ax=plt.subplot(2,1,2)
	
	ax.plot(time,np.sum(np.abs(prob),axis=1))
	# ~ ax.set_xlim(
	
	plt.show()
	
	####################################################################
	
	# ~ # BLOCH
	ax=plt.gca()		
	ax.set_xlim(0,np.max(time))
	ax.plot(time,-xm,c="blue",label=r"Valeur moyenne du site occupé $\langle n \rangle$")
	ax.scatter(tb.beta*h/(F*2*np.pi),(tb.qEs-tb.qEs[int(tb.Ncell/2)])/F,c="red",label=r"Quasi-energie$/F$")
	# ~ ax.scatter((tb.beta+2*np.pi)**h/(F*2*np.pi),(tb.qEs/F-tb.qEs[int(Ncell/2)]/F),c="red",label=r"Quasi-energie$/F$")
	ax.grid()
	
	ax.set_xlabel(r"Temps$*F/\hbar$ (bleue) ou Quasi-moment $\beta$ (rouge)")
	ax.set_title(r"$F=$"+str(F))

	ax.legend()
	plt.show()
	
	
		
	# ~ # BLOCH
	# ~ ax=plt.gca()		
	# ~ ax.set_xlim(0,2*np.pi)
	# ~ ax.plot(F*time*2*np.pi/h,-xm,c="blue",label=r"Valeur moyenne du site occupé $\langle n \rangle$")
	# ~ ax.plot(F*time*2*np.pi/h,-(xmeanTB-xmeanTB[0]),c="green",label=r"Valeur moyenne du site occupé $\langle n \rangle$")
	# ~ ax.scatter(tb.beta,tb.qEs/F-tb.qEs[0]/F,c="red",label=r"Quasi-energie$/F$")
	# ~ ax.scatter(tb.beta+2*np.pi,tb.qEs/F-tb.qEs[0]/F,c="red",label=r"Quasi-energie$/F$")
	# ~ ax.grid()
	
	# ~ ax.set_xlabel(r"Temps$*F/\hbar$ (bleue) ou Quasi-moment $\beta$ (rouge)")
	# ~ ax.set_title(r"$F=$"+str(F))

	# ~ ax.legend()
	# ~ plt.show()
	

if mode=="plotgamma":
	h=1/2.112
	
	dataTB=sys.argv[2]
	tb=TightBinding(dataTB)

	Vn=tb.Vn
	Vn=np.delete(Vn,0)
	gamma=tb.autocorrelation()
	gammaTF=np.abs(np.fft.rfft(gamma)/gamma.size)
	gammaTF=np.delete(gammaTF,0)

	n=np.arange(Vn.size,dtype=float)+1
	
	print(tb.beta)
	
	
	ax=plt.gca()
	ax.scatter(tb.beta,tb.qEs-tb.qEs[0])
	ax.scatter(tb.beta,gamma)
	plt.show()
	
	ax=plt.gca()
	ax.set_xlabel("n")
	ax.set_ylabel(r"$|V_n|$")
	ax.set_yscale('log')
	ax.scatter(n,np.abs(Vn)**2,label=r"$|Vn|^2$",c="blue")
	ax.scatter(n,gammaTF,label="TF autocorrélation",c="red")

	ax.legend()
	plt.show()

	
	




