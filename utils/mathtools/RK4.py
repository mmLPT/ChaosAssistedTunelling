import numpy as np
	
# this is the usually RK4 algorithm that go from y(t) to y(t+dt)
def RK4(f,y,t,dt):
	k1=f(y,t)
	k2=f(y+dt/2.0*k1,t+dt/2.0)
	k3=f(y+dt/2.0*k2,t+dt/2.0)
	k4=f(y+dt*k3,t+dt)
	return y+dt/6.0*(k1+2*k2+2*k3+k4)
