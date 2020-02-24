import sys
sys.path.insert(0, '..') 
import numpy as np
import matplotlib.pyplot as plt

# ~ ax=plt.subplot(3,1,1)

# ~ for i in ["exp-3"]:
	# ~ data=np.load('tempdata/x-'+i+'.npz')
	# ~ xm=data['xm']
	# ~ time=data['time']
	# ~ a=data['a']
	# ~ data.close()
	# ~ xm=xm-xm[0]
	
	
	
	# ~ ax.scatter(time,xm/(2*np.pi))
	
	
	# ~ fit = np.polyfit(time[0:20],xm[0:20]/(2*np.pi), 1)
	# ~ ax.plot(time,fit[0]*time+fit[1],c="red")
	
	# ~ print(fit[0])

	# ~ ax.set_xlim(0,np.max(time))
	# ~ ax.set_ylabel(r"$\sqrt{<x^2>}$")
	# ~ ax.set_xlabel(r"$periods$")

	
	
# ~ ax=plt.subplot(3,1,2)

# ~ for i in ["exp-3"]:
	# ~ data=np.load('tempdata/x-'+i+'.npz')
	# ~ xm=data['xm']
	# ~ time=data['time']
	# ~ a=data['a']
	# ~ data.close()
	# ~ xm=xm-xm[0]
	

	# ~ ax.set_xlim(0,np.max(time))
	# ~ ax.set_xlabel(r"$periods$")

	# ~ ax.plot(time,a)
	# ~ #ax2.plot(time,b)
	# ~ ax.set_ylabel(r"$<(N-1)/2|\psi>$")
	# ~ #ax2.set_ylim(0.0,1.0)
	

# ~ ax=plt.subplot(3,1,3)

ax=plt.gca()

# ~ for i in ["exp-3"]:
	# ~ data=np.load('tempdata/x-'+i+'.npz')
	# ~ time=data['time']
	# ~ b=data['b']
	# ~ data.close()
	
	# ~ for i in range(0,5):
		# ~ ax.plot(time,b[i]/np.max(b[i]))
	
# ~ for i in ["exp-6","exp-7","exp-8","exp-20"]:
# ~ for i in ["Ncell-151-hm-2d6-N20","Ncell-151-hm-2d6-N10","Ncell-151-hm-2d6-N5","Ncell-151-hm-2d6-N1"]:
# ~ for i in ['2-Ncell-151-hm-4d53-N20','2-Ncell-151-hm-4d53-N40']:
for i in ["Ncell-151-hm-2d88-N1","Ncell-151-hm-2d88-N5","Ncell-151-hm-2d88-N10","Ncell-151-hm-2d88-N20"]:
# ~ for i in ["exp-2","exp-3","exp-1","exp-4"]:
	data=np.load('tempdata/'+i+'.npz')
	time=data['time']
	p=data['p']
	data.close()
	
	ax.plot(time,p,label=i)
plt.legend()

# ~ for i in ["exp-3"]:
	# ~ data=np.load('tempdata/x-'+i+'.npz')
	# ~ x=data['x']
	# ~ wf=data['wf']
	# ~ data.close()
	
	
	# ~ for i in range(0,5):
	# ~ ax.plot(x,np.abs(wf[-1,:])**2)
	
	
	# ~ xmax=max(x)
	
	# ~ ax.set_xticks(np.linspace(-xmax/2,xmax/2,np.ceil(0.5*xmax/np.pi)+1,endpoint=True))
	# ~ ax.set_xticklabels([])
	# ~ ax.set_yticks([])
	

	# ~ ax.set_ylabel(r"$\sqrt{<x^2>}$")
	# ~ ax.set_xlabel(r"$periods$")
	
# ~ ax.grid(which='major', color="red",alpha=1.0)	
# ~ ax.set_xlim(-35*2*np.pi,35*2*np.pi)

plt.show()	

# ~ plt.savefig('tempdata/x.png')




