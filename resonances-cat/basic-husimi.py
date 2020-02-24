import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *


Ncell=1
N=64*Ncell
nstates=10

# Run magique
# ~ e=0.50
# ~ gamma=0.25
# ~ h=1/10.9062
# ~ h=1/10.855
# ~ x0=1.1

# ~ gamma=0.375
# ~ e=0.245
# ~ x0=1.8
# ~ h=1/2.8
# ~ gamma=0.315
# ~ e=0.400
# ~ x0=1.8
# ~ h=1/4.7

gamma=0.230
e=0.60
x0=1.5
h=1/3.3

# ~ gamma=0.225
# ~ e=0.59
# ~ x0=1.6
# ~ h=0.3


# ~ SPSclassfile="tempdata/PP/e0d24-g0d375"
# ~ SPSclassfile="tempdata/PP/e0d40-g0d315"
SPSclassfile="tempdata/PP/e0d60-g0d230"
wdir="tempdata/husimi/"
cmap = plt.cm.get_cmap('Purples')


pot=PotentialMP(e,gamma)
grid=Grid(N,h,xmax=Ncell*2*np.pi)
husimi=Husimi(grid,pmax=N*h,scale=5.0)
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)
wf.setState("coherent",x0=x0,xratio=2.8)

# 1 : on diagonalise Floquet
fo.diagonalize()
ind, overlaps=fo.getOrderedOverlapsWith(wf)


# ~ husimi.save(wf,wdir+"initial")
# ~ husimi.npz2png(wdir+"initial",cmapl=cmap,SPSclassbool=True,SPSclassfile=SPSclassfile,textstr="Initial")		

for i in range(0,nstates):
	
	
	datafile="tempdata/husimi/"+strint(i)
	husimi.save(fo.getEvec(ind[i]),datafile)
	qE=4*np.pi*fo.getQE(ind[i])/(h*2*np.pi)
	
	print(i,"/",nstates,np.abs(overlaps[i])**2,"/",qE,qE+1)

	if fo.getEvec(ind[i]).isSymetricInX():
		cmap = plt.cm.get_cmap('Reds')
	else:
		cmap = plt.cm.get_cmap('Blues')
		
	if np.abs(overlaps[i])**2<0.25:
		cmap = plt.cm.get_cmap('Greens')
		
	textstr="{:.1f}".format(100*np.abs(overlaps[i])**2)+r"$\% \quad qE={:.3f}\pi$".format(qE)+r"$\% \quad qE={:.3f}\pi$".format(qE+1)
		
		
	data=np.load(datafile+".npz")
	rho=data["rho"]
	x=data["x"]
	p=data["p"]
	data.close()
	
	fig = latex.fig(columnwidth=345.0,wf=1.0,hf=2.0/np.pi)
	
	ax=plt.gca()
	ax.set_ylim(-2,2)
	ax.set_xlim(-max(x),max(x))
	ax.set_aspect('equal')
	ax.set_xticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi])
	ax.set_xticklabels([r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
	ax.set_yticks([-0.5*np.pi,0,0.5*np.pi])
	ax.set_yticklabels([r"$-\pi/2$",r"$0$",r"$\pi/2$"])
	
	ax.set_xlabel(r"Position")
	ax.set_ylabel(r"Vitesse")
	# ~ ax.set_yticks([])
	# ~ ax.set_yticklabels([])
	# ~ ax.set_xticks([])
	# ~ ax.set_xticklabels([])
	
	img=mpimg.imread(SPSclassfile+".png")
	ax.imshow(img,extent=[-np.pi,np.pi,-2.0, 2.0])
	
	
	mnlvl=0.0
	levels = np.linspace(mnlvl,1.0,100,endpoint=True)	
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
	cf=plt.contourf(x,p,rho, levels=levels,cmap=cmap,alpha=0.5)
	levels = np.linspace(mnlvl,1.0,6)	
	
	# ~ ax.set_title(textstr)
		
	# Sauvergarde sans bordure
	fig.set_frameon(False)
	fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	latex.save(datafile,form="png",dpi=500,transparent=False,bbox_inches='')
		
	# ~ latex.save(datafile,form="png",dpi=500,transparent=False,bbox_inches='tight')
	plt.close(fig)
	
		
	# ~ husimi.npz2png("tempdata/husimi/"+strint(i),cmapl=cmap,SPSclassbool=True,SPSclassfile=SPSclassfile,textstr="{:.1f}".format(100*np.abs(overlaps[i])**2)+r"$\% \quad qE={:.3f}\pi$".format(qE)+r"$\% \quad qE={:.3f}\pi$".format(qE+1))
		









