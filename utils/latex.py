import matplotlib as mpl
import matplotlib.pyplot as plt

# This class is used for nice latex compatible outpout

# ~ fig = latex.fig(columnwidth=345.0)
# ~ latex.save("example")

def fig(columnwidth=500.484, wf=1.0, hf=1.0/1.618):
	# Retourne une figure de taille relative wf en largueur et wf*hf en hauteur
	# La taille relative est calculée par rapport à la largeur du document latex
	# Accessible par la commande \the\columnwidth
	
	# Pour que matplotlib puisse génerer du texte en latex de même police que le .tex
	mpl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}",r'\usepackage{amsmath}',r'\usepackage{amssymb}']
	mpl.rcParams['text.usetex'] = True
	mpl.rcParams['font.family'] = 'lmodern'
	mpl.rcParams['font.size'] = 10
	mpl.rcParams['text.latex.unicode']=True
	
	# Calcule les dimensions de la figure matplotlib par rapport à la largeur du .tex
	fig_width_pt = columnwidth*wf
	inches_per_pt = 1.0/72.27
	fig_width = fig_width_pt*inches_per_pt
	fig_height = fig_width*hf
	
	# Retourne la figure
	return plt.figure(figsize=[fig_width,fig_height])

def save(dfile,form="pdf",dpi=500,transparent=False,bbox_inches='tight'):
	# Sauve la figure en .eps
	if bbox_inches=='tight':
		plt.savefig(dfile+"."+form,dpi=dpi,bbox_inches='tight',format=form,transparent=transparent)
	else:
		plt.savefig(dfile+"."+form,dpi=dpi,format=form,transparent=transparent)
	plt.clf() 
