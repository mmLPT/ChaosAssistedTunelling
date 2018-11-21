import numpy as np

# this is a scheme for SImpson integration
def simpsonIntegrate(npoints,xi,xf,f):
	I=0.0
	dx=abs(xf-xi)/npoints
	for i in range(0,npoints):
		a=xi+i*dx
		b=a+dx
		I=I+dx/6.0*(f(a)+4*f(0.5*(a+b))+f(b))
	return I
