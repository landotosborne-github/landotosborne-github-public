#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from numba import jit
import curses


#class for generating heat system
#######################################################################################################
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


#Define shape array 
##SHAPE MUST BE DEFINED BEFORE init_sys() to set object temp
#THE SIZE SHAPE PIXELS MUST BE SQUARE AND WHOLE FACTOR OF GRID SIZE
#OR ELSE NO SHAPE WILL BE SET
	def get_shape(self, pathtoshape):
		im = np.mean(np.array(Image.open(pathtoshape)),axis=2)
		is_ob = im<100
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
		gtot[0][self.is_ob] = self.ob_temp
		
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
		return self.dt <= check
		
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
def doub_plot(gtot, x_size, grid):
	frames = len(gtot)
	gx = np.linspace(0, x_size , grid)
	gy = np.linspace(0, x_size , grid)
	gx , gy = np.meshgrid(gx, gy)

	f1 = plt.figure(figsize = (15,10))
	ax1 = f1.add_subplot(1,2,1,projection='3d')
	ax1.plot_surface(gx, gy, gtot[frames-1], cmap='inferno')
	ax1.set_zlabel('Temperature (K)')
	ax1.set_title('Surface plot of final system temperature')
	
	ax2 = f1.add_subplot(1,2,2)
	k = ax2.imshow(gtot[frames-1] , cmap='inferno', extent=[0,grid,0,grid])
	ax2.set_title('Heatmap of Final system temperature')
	
	f1.colorbar(k, ax=ax2)
	
	plt.subplots_adjust(wspace=0.25)
	f1.show()


# Create colormap animation of system
def anim_plot(gtot, grid, is_ob):
	frames=len(gtot)
	f = plt.figure(figsize = (15,10))
	ax1 = f.add_subplot(1,2,1)
	a = ax1.imshow(gtot[0] , cmap='inferno', extent=[0,grid,0,grid])
	ax1.set_title('System Temperature over time')
	f.colorbar(a, ax=ax1)
	
	ax2 = f.add_subplot(1,3,3)
	ax2.plot(np.linspace(0,frames-1,frames),np.array([g[is_ob].mean() for g in gtot]))
	ax2.set_xlabel('Time (frames)')
	ax2.set_ylabel('Average Object Temperature (K)')
	ax2.set_title('Avg. Object Temperature through runtime')
	
	def update(i):
		a.set_data(gtot[i,:,:])
		return a,

	ani = animation.FuncAnimation(f,update,frames=len(gtot), interval=20, blit=True)

	plt.subplots_adjust(wspace=0.1)
	f.show()
	return ani


#########################################################################################
#End of simulation functions

#########################################################################################
#Curses Functions

#Creates a input box for user to input integer values including zero
def inputbox_int(stdscr,val):
	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,50,5,10)
		inp.box()
		blurb = 'Enter ' + val +':'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		var = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		try:
			var = int(var)
			if var < 0:
				int('a')
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type. Try again!')
			stdscr.refresh()
			inp.refresh()
			stdscr.getch()
	return var


#Creates a input box for user to input positive float values
def inputbox_float(stdscr,val):
	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,50,5,10)
		inp.box()
		blurb = 'Enter ' + val +':'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		var = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		try:
			var = float(var)
			if var <= 0:
				int('a')
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type. Try again!')
			stdscr.refresh()
			inp.refresh()
			stdscr.getch()
	return var



#Create a input box to enter positive int values excludes zero
def inputbox_intreal(stdscr,val):
	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,50,5,10)
		inp.box()
		blurb = 'Enter ' + val +':'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		var = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		try:
			var = int(var)
			if var <= 0:
				int('a')
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type. Try again!')
			stdscr.refresh()
			inp.refresh()
			stdscr.getch()
	return var

#creates a input box to enter positive integer values must be less than 10000
#Use on grid input to prevent memory overflow
def inputbox_intreal_bound(stdscr,val):
	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,50,5,10)
		inp.box()
		blurb = 'Enter ' + val +':'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		var = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		try:
			var = int(var)
			if var <= 0 or var > 10000:
				int('a')
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type. Try again!')
			stdscr.refresh()
			inp.refresh()
			stdscr.getch()
	return var


def inputbox_resolution(stdscr,val):
	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,50,5,10)
		inp.box()
		blurb = 'Enter ' + val +':'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		var = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		try:
			var = int(var)
			if var <= 0 or var > 10000 or var%100:
				int('a')
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type or format. Try again!')
			stdscr.refresh()
			inp.refresh()
			stdscr.getch()
	return var

#Input box specific to simulation framerate
#extra error handling for runtime%framerate
def inputbox_rate(stdscr,val, runtime):
	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,50,5,10)
		inp.box()
		blurb = 'Enter ' + val +':'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		var = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		try:
			var = int(var)
			if var <= 0 or var > 10000 or runtime%var:
				int('a')
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type or format. Try again!')
			stdscr.refresh()
			inp.refresh()
			stdscr.getch()
	return var

#Creates a menu box for user to select an option
#returns number of chosen option
def menubox(stdscr, start_menu):
	longest_len = len(max(start_menu,key=len))
	start_win = curses.newwin(len(start_menu)+2,longest_len+10,3,2)
	start_win.box()
	start_win.addstr(0,2,"Select Option")
	for i, item in enumerate(start_menu):
		start_win.addstr(i+1,2,item)

	start_win.attron(curses.color_pair(1))
	start_win.addstr(1,2, start_menu[0])
	start_win.attroff(curses.color_pair(1))

	stdscr.refresh()
	start_win.refresh()


	while True:
		key = stdscr.getch()

		# move highlighted menu item up
		if key == curses.KEY_UP:
			current_pos = start_win.getyx()[0]
			if current_pos > 1:
				start_win.addstr(current_pos, 2, start_menu[current_pos-1])
				start_win.attron(curses.color_pair(1))
				start_win.addstr(current_pos-1, 2, start_menu[current_pos-2])
				start_win.attroff(curses.color_pair(1))                		
				start_win.move(current_pos-1, 2)
				start_win.refresh()

		# move highlighted menu item down
		elif key == curses.KEY_DOWN:
			current_pos = start_win.getyx()[0]
			if current_pos < len(start_menu):
				start_win.addstr(current_pos, 2, start_menu[current_pos-1])
				start_win.attron(curses.color_pair(1))
				start_win.addstr(current_pos+1, 2, start_menu[current_pos])
				start_win.attroff(curses.color_pair(1))
				start_win.move(current_pos+1, 2)
				start_win.refresh()

		# select highlighted menu item
		elif key == curses.KEY_ENTER or key == ord("\n"):
			current_pos = start_win.getyx()[0]
			stdscr.clear()
			stdscr.refresh()
			return current_pos
			break



###yes or no input box
#returns True for y or Y else False
def yesnobox(stdscr):
	curses.curs_set(1)
	inp = curses.newwin(3,10,5,30)
	inp.box()
	blurb = 'y/n?'
	inp.addstr(1,1,blurb)
	stdscr.refresh()
	curses.echo()
	a = len(blurb)+2
	conf = inp.getstr(1,a).decode('utf-8')
	curses.noecho()
	if conf =='Y' or conf =='y' :
		return True
	else:
		return False

#Creates a continue box
def contin(stdscr):
	cont = curses.newwin(3,40,5,10)
	cont.box()
	cont.addstr(1,1,'Continue')
	cont.refresh()
	stdscr.refresh()
	stdscr.getch()

#Creates a return to main menu box
#Returns True if wants to return, False otherwise
def ret_main(stdscr):
	stdscr.clear()
	curses.curs_set(1)
	inp = curses.newwin(3,50,5,10)
	inp.box()
	blurb = 'Return to main menu? (y/n):'
	inp.addstr(1,1,blurb)
	stdscr.refresh()
	curses.echo()
	a = len(blurb)+2
	var = inp.getstr(1,a).decode('utf-8')
	curses.noecho()
	if var == 'y' or var == 'Y':
		return True
	else:
		return False


def input_settings(stdscr):
	stdscr.clear()
	stdscr.refresh()
	op = 0
	op = menubox(stdscr, ['Default Settings', 'Customized'])
	
	if op ==1:
		choose_shape=0
		stdscr.addstr(0,0,'Choose Object shape')
		choose_shape = menubox(stdscr, ['None','Circle','Square','Hexagon','Cross', '9_star','Paw'])
		if choose_shape==1:
			pathtoshape='./shapes/project_blank.png'
		elif choose_shape==2:
			pathtoshape='./shapes/project_circle.png'
		elif choose_shape==3:
			pathtoshape='./shapes/project_square.png'
		elif choose_shape==4:
			pathtoshape='./shapes/project_hexagon.png'
		elif choose_shape==5:
			pathtoshape='./shapes/project_cross.png'
		elif choose_shape==6:
			pathtoshape='./shapes/project_9star.png'
		elif choose_shape==7:
			pathtoshape='./shapes/project_paw.png'
			
		runtime = 1000
		l = 100
		rate = 10
		top = 1
		right = 1
		bottom = 1
		left = 1
		bk_tp = 0.000001
		ob_tp = 0.000001
		bk_temp = 0
		ob_temp = 1
	else:
		choose_shape=0
		stdscr.addstr(0,0,'Choose Object shape')
		choose_shape = menubox(stdscr, ['None','Circle','Square','Hexagon','Cross', '9_star','Paw'])
		if choose_shape==1:
			pathtoshape='./shapes/project_blank.png'
		elif choose_shape==2:
			pathtoshape='./shapes/project_circle.png'
		elif choose_shape==3:
			pathtoshape='./shapes/project_square.png'
		elif choose_shape==4:
			pathtoshape='./shapes/project_hexagon.png'
		elif choose_shape==5:
			pathtoshape='./shapes/project_cross.png'
		elif choose_shape==6:
			pathtoshape='./shapes/project_9star.png'
		elif choose_shape==7:
			pathtoshape='./shapes/project_paw.png'
		
		
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: positive integer')
		stdscr.addstr(1,0,'Recommended: ~1000')
		stdscr.addstr(2,0,'Warning: Large inputs lead to longer load times!')
		runtime = inputbox_intreal(stdscr, 'runtime')

		stdscr.clear()
		stdscr.addstr(0,0,'Format: positive integer (less than 10,000 and multiple of 100)')
		stdscr.addstr(1,0,'Recommended: 100')
		stdscr.addstr(2,0,'Warning: Large inputs lead to longer load times!')
		l = inputbox_resolution(stdscr, 'resolution')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: positive integer factor of runtime')
		stdscr.addstr(1,0,'Recommended: ~10')
		stdscr.addstr(2,0,'Warning: Must be a factor of runtime!\nLow framerate means longer animation!')
		rate = inputbox_rate(stdscr, 'framerate',runtime)
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		top = inputbox_int(stdscr, 'top boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		right = inputbox_int(stdscr, 'right boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		bottom = inputbox_int(stdscr, 'bottom boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		left = inputbox_int(stdscr, 'left boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: float')
		stdscr.addstr(1,0,'Recommended: 1e-6')
		stdscr.addstr(2,0,'Warning: Must pass system stability check!')
		bk_tp = inputbox_float(stdscr, 'background thermal coeffiecient')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: float')
		stdscr.addstr(1,0,'Recommended: 1e-6')
		stdscr.addstr(2,0,'Warning: Must pass system stability check!')
		ob_tp = inputbox_float(stdscr, 'object thermal coeffiecient')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'Warning: Larger inputs may obfuscate cmap!')
		bk_temp = inputbox_int(stdscr, 'background temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'Warning: Larger inputs may obfuscate cmap!')
		ob_temp = inputbox_int(stdscr, 'object temperature')

	return pathtoshape, runtime, l, rate, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp



#confirm inputs screen
#just prints inputs and reccomended values to screen
def check_inputs(stdscr,pathtoshape, runtime, l, rate, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp):
	
	stdscr.addstr(1,0,'Do these settings look right?\n',curses.A_BOLD)
	
	s0='Shape : ' + pathtoshape
	stdscr.addstr(2,0,s0)
	
	s1='Runtime : ' + str(runtime)
	stdscr.addstr(3,0,s1)
	stdscr.addstr(3,50,'Recommended: ~1000')
	
	s2='Grid Length : ' + str(l)
	stdscr.addstr(4,0,s2)
	stdscr.addstr(4,50,'Recommended: ~100')
	
	s3='Framerate : ' + str(rate)
	stdscr.addstr(5,0,s3)
	stdscr.addstr(5,50,'Recommended: 5-10')
	
	s4='Top : ' + str(top)
	stdscr.addstr(6,0,s4)
	stdscr.addstr(6,50,'Recommended: 0-100')
	
	s5='Right : ' + str(right)
	stdscr.addstr(7,0,s5)
	stdscr.addstr(7,50,'Recommended: 0-100')
	
	s6='Bottom : ' + str(bottom)
	stdscr.addstr(8,0,s6)
	stdscr.addstr(8,50,'Recommended: 0-100')
	
	s7='Left : ' + str(left)
	stdscr.addstr(9,0,s7)
	stdscr.addstr(9,50,'Recommended: 0-100')
	
	s8='Background thermal coeff. : ' + str(bk_tp)
	stdscr.addstr(10,0,s8)
	stdscr.addstr(10,50,'Recommended: ~1e-6')
	
	s9='Object thermal coeff. : ' + str(ob_tp)
	stdscr.addstr(11,0,s9)
	stdscr.addstr(11,50,'Recommended: ~1e-6')
	
	s10='Background temp : ' + str(bk_temp)
	stdscr.addstr(12,0,s10)
	stdscr.addstr(12,50,'Recommended: 0-100')
	
	s11='Object temp : ' + str(ob_temp)
	stdscr.addstr(13,0,s11)
	stdscr.addstr(13,50,'Recommended: 0-100')
	
	



def input_tests(stdscr, heat_system):
	check = 0
	place = 2
	curses.curs_set(0)
	if heat_system.tp_test(heat_system.bk_tp):
		stdscr.attron(curses.color_pair(2))
		stdscr.addstr(place,0,'PASS [1/5]: Background Thermal coeffiecient passed stability test')
		stdscr.attroff(curses.color_pair(2))
		stdscr.refresh()
		place+=1
	else:
		stdscr.attron(curses.color_pair(3))
		stdscr.addstr(place,0,'FAIL [1/5]: Background Thermal coeffiecient failed stability test! ')
		stdscr.attroff(curses.color_pair(3))
		stdscr.refresh()
		check+=1
		place+=1
		pass
		
	if heat_system.tp_test(heat_system.ob_tp):
		stdscr.attron(curses.color_pair(2))
		stdscr.addstr(place,0,'PASS [2/5]: Object Thermal coeffiecient passed stability test')
		stdscr.attroff(curses.color_pair(2))
		stdscr.refresh()
		place+=1
		pass
	else:
		stdscr.attron(curses.color_pair(3))
		stdscr.addstr(place,0,'FAIL [2/5]: Object Thermal coeffiecient failed stability test')
		stdscr.attroff(curses.color_pair(3))
		stdscr.refresh()
		check+=1
		place+=1
		pass
	
	if heat_system.res_test():
		stdscr.attron(curses.color_pair(2))
		stdscr.addstr(place,0,'PASS [3/5]: Resolution passed modulus test')
		stdscr.attroff(curses.color_pair(2))
		stdscr.refresh()
		place+=1
	else:
		stdscr.attron(curses.color_pair(3))
		stdscr.addstr(place,0,'FAIL[3/5]: Resolution failed modulus test')
		stdscr.attroff(curses.color_pair(3))
		stdscr.refresh()
		check+=1
		place+=1
		pass
	
	if heat_system.rate_test():
		stdscr.attron(curses.color_pair(2))
		stdscr.addstr(place,0,'PASS [4/5]: Framerate passed modulus test')
		stdscr.attroff(curses.color_pair(2))
		stdscr.refresh()
		place+=1
	else:
		stdscr.attron(curses.color_pair(3))
		stdscr.addstr(place,0,'FAIL [4/5]: Framerate failed modulus test')
		stdscr.attroff(curses.color_pair(3))
		stdscr.refresh()
		check+=1
		place+=1
		pass
	
	if heat_system.size_test():
		stdscr.attron(curses.color_pair(2))
		stdscr.addstr(place,0,'PASS [5/5]: Inputs passed size test')
		stdscr.attroff(curses.color_pair(2))
		stdscr.refresh()
		place+=1
	else:
		stdscr.attron(curses.color_pair(3))
		stdscr.addstr(place,0,'FAIL [5/5]: Inputs failed size test')
		stdscr.attroff(curses.color_pair(3))
		stdscr.refresh()
		check+=1
		place+=1
		pass
	
	if check>0:
		stdscr.addstr(0,0,'Tests failed ... Press <enter> to reset',curses.A_BOLD)
		stdscr.getch()
		return False
	else:
		stdscr.addstr(0,0,'Tests passed ... Press <enter> to continue',curses.A_BOLD)
		stdscr.getch()
		return True
	


def input_settings_step(stdscr):
	stdscr.clear()
	stdscr.refresh()
	op = 0
	op = menubox(stdscr, ['Default Settings', 'Customized'])
	
	if op ==1:
		choose_shape=0
		stdscr.addstr(0,0,'Choose Object shape')
		choose_shape = menubox(stdscr, ['None','Circle','Square','Hexagon','Cross', '9_star','Paw'])
		if choose_shape==1:
			pathtoshape='./shapes/project_blank.png'
		elif choose_shape==2:
			pathtoshape='./shapes/project_circle.png'
		elif choose_shape==3:
			pathtoshape='./shapes/project_square.png'
		elif choose_shape==4:
			pathtoshape='./shapes/project_hexagon.png'
		elif choose_shape==5:
			pathtoshape='./shapes/project_cross.png'
		elif choose_shape==6:
			pathtoshape='./shapes/project_9star.png'
		elif choose_shape==7:
			pathtoshape='./shapes/project_paw.png'
			
		step_size = 500
		l = 100
		top = 1
		right = 1
		bottom = 1
		left = 1
		bk_tp = 0.000001
		ob_tp = 0.000001
		bk_temp = 0
		ob_temp = 1
	else:
		choose_shape=0
		stdscr.addstr(0,0,'Choose Object shape')
		choose_shape = menubox(stdscr, ['None','Circle','Square','Hexagon','Cross', '9_star','Paw'])
		if choose_shape==1:
			pathtoshape='./shapes/project_blank.png'
		elif choose_shape==2:
			pathtoshape='./shapes/project_circle.png'
		elif choose_shape==3:
			pathtoshape='./shapes/project_square.png'
		elif choose_shape==4:
			pathtoshape='./shapes/project_hexagon.png'
		elif choose_shape==5:
			pathtoshape='./shapes/project_cross.png'
		elif choose_shape==6:
			pathtoshape='./shapes/project_9star.png'
		elif choose_shape==7:
			pathtoshape='./shapes/project_paw.png'
		
		
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: positive integer')
		stdscr.addstr(1,0,'Recommended: 1-1000')
		stdscr.addstr(2,0,'Warning: Large inputs lead to longer load times!')
		step_size = inputbox_intreal(stdscr, 'step size')

		stdscr.clear()
		stdscr.addstr(0,0,'Format: positive integer (less than 10,000 and multiple of 100)')
		stdscr.addstr(1,0,'Recommended: 100')
		stdscr.addstr(2,0,'Warning: Large inputs lead to longer load times!')
		l = inputbox_resolution(stdscr, 'resolution')
		

		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		top = inputbox_int(stdscr, 'top boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		right = inputbox_int(stdscr, 'right boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		bottom = inputbox_int(stdscr, 'bottom boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'')
		left = inputbox_int(stdscr, 'left boundary temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: float')
		stdscr.addstr(1,0,'Recommended: 1e-6')
		stdscr.addstr(2,0,'Warning: Must pass system stability check!')
		bk_tp = inputbox_float(stdscr, 'background thermal coeffiecient')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: float')
		stdscr.addstr(1,0,'Recommended: 1e-6')
		stdscr.addstr(2,0,'Warning: Must pass system stability check!')
		ob_tp = inputbox_float(stdscr, 'object thermal coeffiecient')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'Warning: Larger inputs may obfuscate cmap!')
		bk_temp = inputbox_int(stdscr, 'background temperature')
		
		stdscr.clear()
		stdscr.addstr(0,0,'Format: integer')
		stdscr.addstr(1,0,'Recommended: 0-100')
		stdscr.addstr(2,0,'Warning: Larger inputs may obfuscate cmap!')
		ob_temp = inputbox_int(stdscr, 'object temperature')

	return pathtoshape, step_size, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp


def check_inputs_step(stdscr,pathtoshape, step_size, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp):
	
	stdscr.addstr(1,0,'Do these settings look right?\n',curses.A_BOLD)
	
	s0='Shape : ' + pathtoshape
	stdscr.addstr(2,0,s0)
	
	s1='Step size : ' + str(step_size)
	stdscr.addstr(3,0,s1)
	stdscr.addstr(3,50,'Recommended: 1-1000')
	
	s2='Grid Length : ' + str(l)
	stdscr.addstr(4,0,s2)
	stdscr.addstr(4,50,'Recommended: ~100')
	
	s4='Top : ' + str(top)
	stdscr.addstr(5,0,s4)
	stdscr.addstr(5,50,'Recommended: 0-100')
	
	s5='Right : ' + str(right)
	stdscr.addstr(6,0,s5)
	stdscr.addstr(6,50,'Recommended: 0-100')
	
	s6='Bottom : ' + str(bottom)
	stdscr.addstr(7,0,s6)
	stdscr.addstr(7,50,'Recommended: 0-100')
	
	s7='Left : ' + str(left)
	stdscr.addstr(8,0,s7)
	stdscr.addstr(8,50,'Recommended: 0-100')
	
	s8='Background thermal coeff. : ' + str(bk_tp)
	stdscr.addstr(9,0,s8)
	stdscr.addstr(9,50,'Recommended: ~1e-6')
	
	s9='Object thermal coeff. : ' + str(ob_tp)
	stdscr.addstr(10,0,s9)
	stdscr.addstr(10,50,'Recommended: ~1e-6')
	
	s10='Background temp : ' + str(bk_temp)
	stdscr.addstr(11,0,s10)
	stdscr.addstr(11,50,'Recommended: 0-100')
	
	s11='Object temp : ' + str(ob_temp)
	stdscr.addstr(12,0,s11)
	stdscr.addstr(12,50,'Recommended: 0-100')



#####################################################################################
#end of curses funcitons

#####################################################################################
#DISPLAY SCREEN FUNCTIONS

#Display of Simulation mode 1 
def display1(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to Object Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)

	contin(stdscr)
	
	pathtoshape, runtime, l, rate, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp = input_settings(stdscr)
		
	stdscr.clear()
	stdscr.addstr(0,0,'Object Simulation Mode\n',curses.A_BOLD)
	
	check_inputs(stdscr,pathtoshape, runtime, l, rate, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
	 
	if yesnobox(stdscr):
		stdscr.clear()
		stdscr.refresh()
		
		heat_sys = heat_system(runtime, rate, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
		
		
		if input_tests(stdscr, heat_sys):
			
			heat_sys.get_shape(pathtoshape)
			gtot = heat_sys.init_sys()
			
			stdscr.clear()
			curses.curs_set(0)
			stdscr.addstr(0,0,'Loading ...', curses.A_BOLD)
			stdscr.refresh()
		
			gtot = heat_jump(gtot, heat_sys.dt, heat_sys.dx, heat_sys.bk_tp, heat_sys.runtime, heat_sys.rate, heat_sys.ob_tp , heat_sys.is_ob )
			
			stdscr.clear()
			stdscr.addstr(0,0,'Creating plot and animation ...', curses.A_BOLD)
			stdscr.addstr(1,0,'Please wait', curses.A_BOLD)
			stdscr.refresh()
			doub_plot(gtot, heat_sys.x_size, heat_sys.grid)
			
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue to animation', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			
			plt.show()
			
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue to animation', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			stdscr.getch()
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			ani = anim_plot(gtot, heat_sys.grid, heat_sys.is_ob)
			plt.show()
			contin(stdscr)
		else:
			stdscr.clear()
			stdscr.addstr('Press enter to reset...')
			stdscr.refresh()
			stdscr.getch()
			display1(stdscr)
		
	else:
		stdscr.clear()
		stdscr.addstr('Press enter to reset...')
		stdscr.refresh()
		stdscr.getch()
		display1(stdscr)


	stdscr.clear()
	stdscr.refresh()
	


#Display of Simulation mode 2
def display2(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to Step Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)

	contin(stdscr)
	
	pathtoshape, step_size, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp = input_settings_step(stdscr)
		
	stdscr.clear()
	stdscr.addstr(0,0,'Step Simulation Mode\n',curses.A_BOLD)
	
	check_inputs_step(stdscr,pathtoshape, step_size, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
	 
	if yesnobox(stdscr):
		stdscr.clear()
		stdscr.refresh()
		
		heat_sys = heat_system(1, 1, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
		
		
		if input_tests(stdscr, heat_sys):
			
			heat_sys.get_shape(pathtoshape)
			gtot = heat_sys.init_sys()
			
			
			while True:
				
				stdscr.clear()
				curses.curs_set(0)
				stdscr.addstr(0,0,'Loading ...', curses.A_BOLD)
				stdscr.refresh()
				
				for i in range(step_size):
					gtot[0] = heat_step(gtot[0], heat_sys.dt, heat_sys.dx, heat_sys.bk_tp, heat_sys.ob_tp , heat_sys.is_ob )
			
				stdscr.clear()
				stdscr.addstr(0,0,'Creating plot ...', curses.A_BOLD)
				stdscr.addstr(1,0,'Please wait', curses.A_BOLD)
				stdscr.refresh()
				doub_plot(gtot, heat_sys.x_size, heat_sys.grid)
			
				stdscr.clear()
				stdscr.addstr(0,0,'Plot must be closed to continue', curses.A_BOLD)
				stdscr.addstr(1,0,'Press <enter> to continue')
				stdscr.refresh()
			
				plt.show()
				plt.close('all')
				
				stdscr.addstr(3,0,"Step Again?", curses.A_BOLD)
				stdscr.refresh()
				if not yesnobox(stdscr):
					break
			
		else:
			stdscr.clear()
			stdscr.addstr('Press enter to reset...')
			stdscr.refresh()
			stdscr.getch()
			display1(stdscr)
		
	else:
		stdscr.clear()
		stdscr.addstr('Press enter to reset...')
		stdscr.refresh()
		stdscr.getch()
		display1(stdscr)


	stdscr.clear()
	stdscr.refresh()
	
	


#Display of Simulation mode 3
def display3(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to Sandbox Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)

	contin(stdscr)
	
	pathtoshape, runtime, l, rate, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp = input_settings(stdscr)
		
	stdscr.clear()
	stdscr.addstr(0,0,'Sandbox Mode\n',curses.A_BOLD)
	
	check_inputs(stdscr,pathtoshape, runtime, l, rate, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
	


	if yesnobox(stdscr):
		stdscr.clear()
		stdscr.refresh()
		
		heat_sys = heat_system(runtime, rate, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
		
		
		if input_tests(stdscr, heat_sys):
			heat_sys = heat_system(runtime, rate, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
			heat_sys.get_shape(pathtoshape)
			gtot = heat_sys.init_sys()
			stdscr.clear()
			curses.curs_set(0)
			stdscr.addstr(0,0,'How would you like the internal temperature to be set?')
			
			b = menubox(stdscr,['Random','Abs(Random)','Centered Dot'])
			if b == 1:
				gtot[0, 1:-1, 1:-1] = gtot[0,1:-1,1:-1] + np.random.normal(size = (l-2,l-2))*heat_sys.ob_temp
				pass
			elif b == 2:
				gtot[0, 1:-1, 1:-1] = gtot[0,1:-1,1:-1] + abs(np.random.normal(size = (l-2,l-2))*heat_sys.ob_temp)
				pass
			elif b == 3:
				mi = heat_sys.grid//2 - heat_sys.grid//20
				ma = heat_sys.grid//2 + heat_sys.grid//20
				gtot[0,mi:ma,mi:ma] = heat_sys.ob_temp
				pass
			
			stdscr.clear()
			curses.curs_set(0)
			stdscr.addstr(0,0,'Loading ...', curses.A_BOLD)
			stdscr.refresh()
			
			gtot = heat_jump(gtot, heat_sys.dt, heat_sys.dx, heat_sys.bk_tp, heat_sys.runtime, heat_sys.rate, heat_sys.ob_tp, heat_sys.is_ob)
			
			stdscr.clear()
			stdscr.addstr(0,0,'Creating plot and animation ...', curses.A_BOLD)
			stdscr.addstr(1,0,'Please wait', curses.A_BOLD)
			stdscr.refresh()
			doub_plot(gtot, heat_sys.x_size, heat_sys.grid)
			
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue to animation', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			
			plt.show()
			
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue to animation', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			stdscr.getch()
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			ani = anim_plot(gtot, heat_sys.grid, heat_sys.is_ob)
			plt.show()
			
			contin(stdscr)
		else:
			stdscr.clear()
			stdscr.addstr('Press enter to reset...')
			stdscr.refresh()
			stdscr.getch()
			display3(stdscr)
	else:
		stdscr.clear()
		stdscr.addstr('Press enter to reset...')
		stdscr.refresh()
		stdscr.getch()
		display3(stdscr)
	
	stdscr.clear()
	
    

def help_display(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to the Help page\nPlease look at some of the manuals and FAQs below \n',curses.A_BOLD)

	choose = menubox(stdscr, ['Quick Guide','The Math behind the Simulation','Why do my settings keep failing input tests?','What is the stability test?','Why do I get stuck when the plots open?','What is "invalid command name "2873334728_on_timer""?'])
	
	if choose==1:#quick guide
		stdscr.clear()
		stdscr.addstr(0,0,'Quick Guide', curses.A_BOLD)
		stdscr.addstr(1,0,'Press <enter> to reset')
		stdscr.addstr(2,0,'1.) Select a simulation mode\n2.)Choose Default settings or customize your system settings.\n3.)Double check your system settings and enter <y> if they are correct.\n4.)Ensure your settings have passed the system tests. If all the tests pass press <enter> to continue to the simulation plots.\n5.)Wait for the final state plot to print. Once it has printed you can save the file using matplotlib or exit to continue.\n6.) Once all the available plots have been displayed press <y> to return to the main menu or <n> to exit the program.\n7.) If at any point there is difficulty in getting the program to work a certain way confront this help guide or try using the default system settings.\n\n Some notes on how the simulation runs:\n-All Boundary conditions are held constant\n-All internal temperatures evolve via the heat eqn. (aka no internal heat generation)\n\nI hope you enjoy the 2D Heat Equation Simulator!')
		stdscr.refresh()
		stdscr.getch()
		
			

	
	elif choose == 2:#Math behind program
		stdscr.clear()
		stdscr.addstr(0,0,'The Math behind the Simulation', curses.A_BOLD)
		stdscr.addstr(1,0,'Press <enter> to reset')	
		stdscr.addstr(2,0,'This project simulates the 2D heat equation by using the FTCS finite-difference method. This method changes the partial differential equation into a series of discrete differences between points in a grid of temperature values. This allows us to evolve the system forward in time by iterating this process over the grid. It is an explicit method which means that error can build up if not careful, which is why the stability tests ensure only stable simulations can be run. For more information on the heat equation in general visit\nhttps://en.wikipedia.org/wiki/Heat_equation\nor for the specific math used for this program visit\nhttps://en.wikipedia.org/wiki/FTCS_scheme')
		stdscr.refresh()
		stdscr.getch()

	
	elif choose == 3:#Failing input tests
		stdscr.clear()
		stdscr.addstr(0,0,'Why do my settings keep failing input tests?', curses.A_BOLD)
		stdscr.addstr(1,0,'Press <enter> to reset')
		stdscr.addstr(2,0,'Some settings need to be in a certain format or factors of each other for some modes to function properly. Try reading the recommended values or using the Simple/Default simulation mode.\nSome of the most commmon settings that need specific formatting are the Runtime and Framerate, the framerate must be a factor of the runtime and the runtime cannot be too large, the Resolution, which must be a factor of 100, and the thermal coefficients, which must pass the system stability tests')
		stdscr.refresh()
		stdscr.getch()

	
	elif choose == 4:#Stability tests
		stdscr.clear()
		stdscr.addstr(0,0,'What is the stability test?', curses.A_BOLD)
		stdscr.addstr(1,0,'Press <enter> to reset')
		stdscr.addstr(2,0,'The stability test is a test performed on the user inputted thermal coefficients which ensures that the error involved in the finite difference calculations is stable and will remain constant or shrink as the simulation runs. It uses Von-Neumann Stability analysis and for our purposes it means the thermal coefficient times the time differential over the space differential squared must be less than 1/4\n-- a*(dt/dx**2) <= 0.25 --\nFor more information on this visit\nhttps://en.wikipedia.org/wiki/Von_Neumann_stability_analysis')
		stdscr.refresh()
		stdscr.getch()

	
	elif choose == 5:#Stuck at plots open
		stdscr.clear()
		stdscr.addstr(0,0,'Why do I get stuck when the plots open?', curses.A_BOLD)
		stdscr.addstr(1,0,'Press <enter> to reset')
		stdscr.addstr(2,0,'In order to progress in the program all matplotlib plots must be closed manually. If you are stuck in the program try closing all plots and animations.')
		stdscr.refresh()
		stdscr.getch()

	
	elif choose == 6:#Animation bug
		stdscr.clear()
		stdscr.addstr(0,0,'What is "invalid command name "2873334728_on_timer""?', curses.A_BOLD)
		stdscr.addstr(1,0,'Press <enter> to reset')
		stdscr.addstr(2,0,'This is a harmless bug that appears because of how the program uses matplotlib plots. \nThis warning is printed to the screen if the matplotlib animation was closed at a point in the middle of its animation.\nDue to how the plots are displayed it is currently unfixable but harmless. \nThe program is written so that it automatically clears this message once the plot in front of it is closed. \nIt is not very aesthetically pleasing, but unfortunately this is the best fix I could find for now. \nApologies!')
		stdscr.refresh()
		stdscr.getch()
		
	

##########################################################################################
#End of Curses Functions


def main(stdscr):
	stdscr = curses.initscr()
	curses.curs_set(0)
	curses.start_color()
	curses.init_pair(1,curses.COLOR_BLACK, curses.COLOR_GREEN)
	curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
	curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
	stdscr.clear()

	welcome= "Welcome to the 2D Heat Equation Simulator! v1.2"
	stdscr.addstr(0,0,welcome,curses.A_BOLD)

	start_menu=["Simulation Mode","Step Mode","Sandbox Mode", "Help", "Exit"]

	a = menubox(stdscr, start_menu)
    
	if a==1:#DISPLAY1
		display1(stdscr)
		if ret_main(stdscr):
			main(stdscr)
		else:
			stdscr.clear()
			stdscr.refresh()
	elif a==2:#DISPLAY2
		display2(stdscr)
		if ret_main(stdscr):
			main(stdscr)
		else:
			stdscr.clear()
			stdscr.refresh()
	elif a==3:#DISPLAY3
		display3(stdscr)
		if ret_main(stdscr):
			main(stdscr)
		else:
			stdscr.clear()
			stdscr.refresh()
	elif a==4:#HELP SCREEN
		 help_display(stdscr)
		 if ret_main(stdscr):
			 main(stdscr)
		 else:
			 stdscr.clear()
			 stdscr.refresh()
	elif a==5:#QUIT
		 pass

	stdscr.clear()
	stdscr.addstr(0,0, 'Thanks for using the Simulation!', curses.A_BOLD)
	stdscr.refresh()
	curses.curs_set(1)
	#stdscr.getch() #These getchs accumulate for every ret_main need to fix
	# cleanup curses
	curses.endwin()

	
		
	



curses.wrapper(main)


