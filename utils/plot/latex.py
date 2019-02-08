import matplotlib as mpl
import matplotlib.pyplot as plt

# This class is used for nice latex compatible outpout


#~ lw=0.7
#~ fs=10
#~ props = dict(boxstyle='square', facecolor='white', alpha=1.0)	

# The following function makes give size 
def get_figsize(columnwidth=500.484, wf=1.0, hf=1.0):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

def setLatex():
	mpl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}",r'\usepackage{amsmath}',r'\usepackage{amssymb}']
	mpl.rcParams['text.usetex'] = True
	mpl.rcParams['font.family'] = 'lmodern'
	mpl.rcParams['font.size'] = 10
	mpl.rcParams['text.latex.unicode']=True
	
def saveLatex(datafile,notes=True):	
	if notes:
		#plt.savefig("/users/martinez/Desktop/These/notes/pictures/"+datafile+".pdf", format="pdf", bbox_inches='tight')
		plt.savefig("/users/martinez/Desktop/These/notes/pictures/"+datafile+".png", bbox_inches='tight',dpi=1000)
	else:
		plt.savefig(datafile+".png", bbox_inches='tight',dpi=1000)
	plt.clf()
