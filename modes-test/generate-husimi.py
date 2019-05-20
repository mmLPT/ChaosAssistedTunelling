import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

from utils.quantum import *
from utils.classical import *
from utils.systems.modulatedpendulum import *
from utils.systems.general import *
from utils.toolsbox import *

n=4
N=64
e=0.44
gamma=0.267
h=0.292
x0=np.pi/2

pot=PotentialMP(e,gamma)
grid=Grid(N,h)
husimi=Husimi(grid)
fo=CATFloquetOperator(grid,pot)
wf=WaveFunction(grid)

# 1 : on diagonalise Floquet
fo.diagonalize()
# 2 : on créé un fonction d'onde DROITE
wf.setState("coherent",x0=x0,xratio=2.0)
# 3 : on calcule le recouvrement entre les vecteurs propres de Floquet et la fonction d'onde DROITE (ainsi que les quasienergies)
fo.computeOverlapsAndQEs(wf)
# 4 : on récupère les n états qui recouvre le plus
# qes : quasienergies
# overlaps : projections
# symX : l'état est-il symétrique en x
# ind : tableau contenant l'index des états
qes, overlaps, symX, ind=fo.getQEsOverlapsSymmetry(n,True)

# 5- pour chacun des n états
for i in range(0,n):
	# 6 - on sauve le husimi 
	husimi.save(fo.getEvec(ind[i],False),str(i))
	husimi.npz2png(str(i))






