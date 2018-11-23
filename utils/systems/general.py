from utils.classical.stroboscopic import *
from utils.classical.timepropagator import *
from utils.quantum.husimi import *
from utils.quantum.wavefunction import *
from utils.quantum.quantumoperator import *
from utils.toolsbox import *
import utils.plot.read as read

# This scripts contains functions that are usable with any system


def classical(pot,nperiod,ny0,datafile):
	# Classical stroboscopic phase
	cp=ClassicalContinueTimePropagator(pot)
	sb=StrobosopicPhaseSpace(nperiod,ny0,cp) #,pmax=np.pi)
	sb.save(datafile)
	sb.npz2plt(datafile)

def propagate( grid, pot, iperiod, icheck,wdir,T0=1,idtmax=1):
	# Propagate a wavefunction over one period/one kick/on length
	# I periodically saves Husimi representation and wf in a .npz file
	husimi=Husimi(grid)
	
	fo=CATFloquetOperator(grid,pot,T0=T0,idtmax=idtmax)

	
	projs=np.zeros(int(iperiod/icheck))
	time=np.zeros(int(iperiod/icheck))
	
	wfcs=WaveFunction(grid)
	wfcs.setState("coherent",x0=pot.x0,xratio=2.0)
	
	wf=WaveFunction(grid)
	wf.setState("coherent",x0=pot.x0, xratio=2.0)
	
	for i in range(0,iperiod):
		if i%icheck==0:
			print(str(i+1)+"/"+str(iperiod))
			time[int(i/icheck)]=i
			husimi.save(wf,wdir+"husimi/"+strint(i/icheck))
			wf.save(wdir+"wf/"+strint(i/icheck))
			projs[int(i/icheck)]=wfcs//wf
		fo.propagate(wf)
	
	np.savez(wdir+"projs","w", time=time, projs=projs)
	read.projs(wdir+"projs")
