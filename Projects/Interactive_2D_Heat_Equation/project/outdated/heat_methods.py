#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from PIL import Image
from numba import jit


#class for generating heat system
#######################################################################################################33
class heat_system(object):
	def __init__(self, runtime , rate, l , top , right , bottom, left, bk_tp, ob_tp, bk_temp, ob_temp, dt=1, x_size=1 ):
		self.runtime = runtime
		self.rate = rate
		self.grid = l
		self.top = top
		self.right = right
		self.bottom = bottom
		self.left = left
		self.bk_tp = bk_tp
		self.ob_tp = ob_tp
		self.bk_temp = bk_temp
		self.ob_temp = ob_temp
		#Assumes that there is no object unless defined with get_shape()
		self.is_ob = np.zeros((self.grid,self.grid),dtype=bool)
		self.dt = dt
		self.x_size = x_size
		self.dx = self.x_size / self.grid


#Define shape array ##SHAPE MUST BE DEFINED BEFORE init_sys() to set object temp
	def get_shape(self, pathtoshape):
		im = np.mean(np.array(Image.open(pathtoshape)),axis=2)
		is_ob = im<10
		if len(is_ob)==len(is_ob[0]): #check pixels are square
			if not (self.grid%len(is_ob)): #check grid is evenly divisible by shape
				scale = self.grid // len(is_ob)
				is_ob = np.repeat(np.repeat(is_ob,scale,axis=0),scale,axis=1)
				self.is_ob = is_ob


#initialize pixels for simulation
	def init_sys(self):
		frames = self.runtime // self.rate
		gtot = np.zeros((frames , self.grid , self.grid))
		
		#set background temp
		gtot[0] = self.bk_temp
		
		#set object temperature
		gtot[0][self.is_ob==1] = self.ob_temp
		
		#set edges 
		gtot[0,:,0] = self.left
		gtot[0,:,-1] = self.right
		gtot[0,0,:] = self.top
		gtot[0,-1,:] = self.bottom
	
		return gtot

########TEST FUNCTIONS
########Return True for pass, False for fail

#Run a stability test for system given specified tp
	def tp_test(self, tp):
		check = (self.dx**2) / (4*tp)
		if self.dt <= check:
			return True
		else:
			return False
		
#tests input resolution
#must be multiple of 100
	def res_test(self):
		return not self.grid % 100
	
#tests input framerate and runtime
#framerate must be factor of runtime and less than runtime
	def rate_test(self):
		return not self.runtime%self.rate

#tests gtot size
#makes sure resulting array is not too large
	def size_test(self):
		max_npsize= 7.5e7
		return max_npsize > ((self.runtime//self.rate)*self.grid*self.grid)

################################################################################################################
#end of system class

#Simulation functions
################################################################################################################

#Calculate all frames of simulation
@jit
def heat_jump(gtot, dt , dx , bk_tp, runtime, rate, ob_tp, is_ob):
	g = gtot[0].copy()
	l = len(g[0])
	t=0
	f=0
	bk_alph = (bk_tp*dt)/dx**2
	ob_alph = (ob_tp*dt)/dx**2
	for t in range(1,runtime):
		ng = g.copy()
		for i in range(1 ,l-1):
			for j in range(1,l-1):
				if is_ob[j][i]:
					ng[j][i] = g[j][i] + (ob_alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[j][i])
					pass
				else:
					ng[j][i] = g[j][i] + (bk_alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[j][i])
					
		g = ng.copy()
		if t % rate ==0:
			f+=1
			gtot[f] = g
		
	return gtot


#Calculate next frame of simulation
@jit
def heat_step(g, dt , dx , bk_tp, ob_tp, is_ob):
	l = len(g[0])
	ng = g.copy()
	bk_alph = (bk_tp*dt)/dx**2
	ob_alph = (ob_tp*dt)/dx**2
	for i in range(1 ,l-1):
		for j in range(1,l-1):
			if is_ob[j][i]:
				ng[j][i] = g[j][i] + (ob_alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[j][i])
			else:
				ng[j][i] = g[j][i] + (bk_alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[j][i])
	g = ng.copy()
	return g




#Create surface plot and colormap of final sim state
def doub_plot(gtot, x_size, grid, is_ob):
	frames = len(gtot)
	gx = np.linspace(0, x_size , grid)
	gy = np.linspace(0, x_size , grid)
	gx , gy = np.meshgrid(gx, gy)

	f1 = plt.figure(figsize = (15,5))
	ax1 = f1.add_subplot(131,projection='3d')
	ax1.plot_surface(gx, gy, gtot[frames-1], cmap='inferno')
	ax2 = f1.add_subplot(132)
	a = ax2.imshow(gtot[frames-1] , cmap='inferno', extent=[0,l,0,l])
	f1.colorbar(a, ax=ax2)
	ax3 = f1.add_subplot(133)
	ax3.plot(np.linspace(0,frames-1,frames),np.array([g[is_ob].mean() for g in gtot]))
	f1.show()


# Create colormap animation of system
def anim_plot(gtot):
	f = plt.figure(figsize = (5,5))
	ax = f.add_subplot(111)
	a = ax.imshow(gtot[0] , cmap='inferno', extent=[0,l,0,l])
	f.colorbar(a, ax=ax)
	
	def update(i):
		a.set_data(gtot[i,:,:])
		return a,

	ani = animation.FuncAnimation(f,update,frames=len(gtot), interval=10, blit=True)
	f.show()
	input('Press <enter> to continue')
	return ani





#########################################################################################
#End of simulation functions

#########################################################################################
#Start Simulation

runtime = 1000
rate = 5
l = 100
top = 0
right = 0
bottom = 0
left = 0
bk_tp = 0.000001
ob_tp = 0.000001
bk_temp = 0
ob_temp = 10



heat_system = heat_system( runtime , rate, l , top , right , bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)

heat_system.get_shape('./shapes/project_hexagon.png')

gtot = heat_system.init_sys()

print('Background stability Check')
if heat_system.tp_test(bk_tp):
	print('Background stable!')
else:
	print('Background unstable!')

print('Object stability Check')
if heat_system.tp_test(ob_tp):
	print('Object stable!')
else:
	print('Object unstable!')


print('Loading ...\nPlease wait')



gtot = heat_jump(gtot, heat_system.dt, heat_system.dx, heat_system.bk_tp, heat_system.runtime, heat_system.rate, heat_system.ob_tp , heat_system.is_ob )

print('Printing final state!')

doub_plot(gtot, heat_system.x_size, heat_system.grid, heat_system.is_ob)

input('Press <enter> to continue to animation\n')

ani = anim_plot(gtot)


input('Press <enter> to exit ...')

