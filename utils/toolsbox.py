import numpy as np


def strint(i):
	return "{:05d}".format(int(i))

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

def convert2theory(s,nu,x0exp=0.0):
	hbar=1.0545718 #e-34
	u=1.660538921 #e-27
	m=86.909180527*u 
	d=532.0 #e-9
	nuL=(np.pi*hbar)/(m*d**2)*10**(11)
	gamma=s*(nuL/nu)**2
	heff=2*(nuL/nu)
	x0=x0exp*np.pi/180
	if x0==0.0:
		return gamma, heff
	else:
		return gamma, heff, x0
	
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
