#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import numba

#grid size
l = 50

#intervals and size
x_size = y_size = 1
dt = 1
dx = x_size / l 
runtime = 100


#thermal properties
tp = 0.000001

#Rate of runtime per frame
rate= 10
frames = runtime // rate

#initialize system
gx = np.linspace(0, x_size , l)
gy = np.linspace(0, y_size , l)
gx , gy = np.meshgrid(gx, gy)

gtot = np.zeros((frames , l , l))

g = np.zeros((l ,l))


#set all
gtot[0] = 1

#set edges 
edg = 10
gtot[0,:,0] = edg
gtot[0,:,-1] = edg
gtot[0,0,:] = edg
gtot[0,-1,:] = edg


#set boundary (1 index in)
#If edges set to 0 kinda simulates open system

gtot[0,1:-2,1] = 1
gtot[0,1:-2,-2] = 1
gtot[0,1,1:-2] = 1
gtot[0,-2,1:-1] = 1


#set pixels

#Random
#gtot[0, 1:-1, 1:-1] = np.random.normal(size = (l-2,l-2))*100

#abs Random
#gtot[0, 1:-1, 1:-1] = abs(np.random.normal(size = (l-2,l-2))*10)

#centered_dot
mi = l//2 - l//20
ma = l//2 + l//20
#gtot[0,mi:ma,mi:ma] = 10


#stability check
check = (dx**2)/(4*tp)
if dt <= check:
	print('Stable!')
	print( dt, ' <= ' , check)
else:
	print('Unstable?!')
	print(check, ' = (dx**2)/(4*tp) should be greater than dt = ' , dt)


@numba.jit
def heat_jump(gtot, dt , dx , tp, runtime, rate):
	g = gtot[0].copy()
	l = len(g[0])
	t=0
	f=0
	alph = (tp*dt)/dx**2
	for t in range(1,runtime):
		ng = g.copy()
		for i in range(1 ,l-1):
			for j in range(1,l-1):
				ng[j][i] = g[j][i] + (alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[j][i])
		g = ng.copy()
		
		if t % rate ==0:
			f+=1
			gtot[f] = g
		
		#gtot[t] = g
		#t+=dt
	return gtot
	
@numba.jit
def heat_step(g, dt , dx , tp):
	ng = g.copy()
	alph = (tp*dt)/dx**2
	for i in range(1 ,l-1):
		for j in range(1,l-1):
			ng[j][i] = g[j][i] + (alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[i][j])
	g = ng.copy()
	return g


print('Loading ...\nPlease wait')

gtot = heat_jump( gtot, dt, dx, tp, runtime, rate)
#gtot[frames-1] = heat_step(gtot[0], dt, dx ,tp)


f1 = plt.figure(figsize = (15,5))

ax1 = f1.add_subplot(121,projection='3d')

ax1.plot_surface(gx, gy, gtot[frames-1], cmap='inferno')

ax2 = f1.add_subplot(122)

a = ax2.imshow(gtot[frames-1] , cmap='inferno', extent=[0,l,0,l])

f1.colorbar(a, ax=ax2)

print('Printing final state!')
f1.show()

input('Press <enter> to continue to animation ...\n')


f2 = plt.figure(figsize = (15,5))

ax = f2.add_subplot(111)

a = ax.imshow(gtot[frames-1] , cmap='inferno', extent=[0,l,0,l])


def update(i):
	a.set_data(gtot[i,:,:])
	return a,

ani = animation.FuncAnimation(f2,update,frames=gtot.shape[0], interval=10, blit=True)

f2.show()

input('Press <enter> to exit ...')

