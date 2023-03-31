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
		gtot[0][self.is_ob] = self.ob_temp
		
		#set edges 
		gtot[0,:,0] = self.left
		gtot[0,:,-1] = self.right
		gtot[0,0,:] = self.top
		gtot[0,-1,:] = self.bottom
	
		return gtot


#Run a stability check for system given specified tp
	def stability_check(self, tp):
		check = (self.dx**2) / (4*tp)
		if self.dt <= check:
			return True
		else:
			return False


		

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

	f1 = plt.figure(figsize = (15,5))
	ax1 = f1.add_subplot(121,projection='3d')
	ax1.plot_surface(gx, gy, gtot[frames-1], cmap='inferno')
	ax2 = f1.add_subplot(122)
	k = ax2.imshow(gtot[frames-1] , cmap='inferno', extent=[0,grid,0,grid])
	f1.colorbar(k, ax=ax2)
	f1.show()


# Create colormap animation of system
def anim_plot(gtot, grid):
	f = plt.figure(figsize = (5,5))
	ax1 = f.add_subplot(111)
	a = ax1.imshow(gtot[0] , cmap='inferno', extent=[0,grid,0,grid])
	f.colorbar(a, ax=ax1)
	
	def update(i):
		a.set_data(gtot[i,:,:])
		return a,

	ani = animation.FuncAnimation(f,update,frames=len(gtot), interval=10, blit=True)
	f.show()
	return ani

#########################################################################################
#End of simulation functions

#########################################################################################
#Curses Functions

#Creates a input box for user to input positive integer values
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


#Creates a menu box for user to select an option
#returns number of chosen option
def menubox(stdscr, start_menu):
	start_win = curses.newwin(len(start_menu)+2,40,3,2)
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
			if current_pos == 1:
				stdscr.clear()
				stdscr.refresh()
				return 1
				#option 1 code
				break
			elif current_pos == 2:
				#option 2 code
				stdscr.clear()
				stdscr.refresh()
				return 2
				break
			elif current_pos == 3:
				stdscr.clear()
				stdscr.refresh()
				return 3
				break



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
		runtime = 1000
		l = 100
		rate = 10
		top = 0
		right = 0
		bottom = 0
		left = 0
		bk_tp = 0.000001
		ob_tp = 0.000001
		bk_temp = 0
		ob_temp = 1
		pathtoshape='./shapes/project_hexagon.png'#####SET THIS TO project_blank.png
	else:
		choose_shape=0
		choose_shape = menubox(stdscr, ['Square','Hexagon','Cross', '9_star'])
		if choose_shape==1:
			pathtoshape='./shapes/project_square.png'
		elif choose_shape==2:
			pathtoshape='./shapes/project_hexagon.png'
		elif choose_shape==3:
			pathtoshape='./shapes/project_cross.png'
		elif choose_shape==4:
			pathtoshape='./shapes/project_9star.png'
		
		pathtoshape='./shapes/project_hexagon.png'
		
		runtime = inputbox_int(stdscr, 'runtime')
		
		l = inputbox_int(stdscr, 'resolution')
		
		rate = inputbox_int(stdscr, 'framerate')
		
		top = inputbox_int(stdscr, 'top boundary temperature')
		
		right = inputbox_int(stdscr, 'right boundary temperature')
		
		bottom = inputbox_int(stdscr, 'bottom boundary temperature')
		
		left = inputbox_int(stdscr, 'left boundary temperature')
		
		bk_tp = inputbox_float(stdscr, 'background thermal coeffiecient')
		
		ob_tp = inputbox_float(stdscr, 'object thermal coeffiecient')
		
		bk_temp = inputbox_int(stdscr, 'background temperature')
		
		ob_temp = inputbox_int(stdscr, 'object temperature')



#Display of Simulation mode 1 
def display1(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to Object Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)

	contin(stdscr)
	stdscr.clear()
	stdscr.refresh()
	op = 0
	op = menubox(stdscr, ['Default Settings', 'Customized'])

	if op ==1:
		runtime = 1000
		l = 100
		rate = 10
		top = 0
		right = 0
		bottom = 0
		left = 0
		bk_tp = 0.000001
		ob_tp = 0.000001
		bk_temp = 0
		ob_temp = 1
		pathtoshape='./shapes/project_hexagon.png'#####SET THIS TO project_blank.png
	else:
		pathtoshape='./shapes/project_hexagon.png'
		
		runtime = inputbox_int(stdscr, 'runtime')
		
		l = inputbox_int(stdscr, 'resolution')
		
		rate = inputbox_int(stdscr, 'framerate')
		
		top = inputbox_int(stdscr, 'top boundary temperature')
		
		right = inputbox_int(stdscr, 'right boundary temperature')
		
		bottom = inputbox_int(stdscr, 'bottom boundary temperature')
		
		left = inputbox_int(stdscr, 'left boundary temperature')
		
		bk_tp = inputbox_float(stdscr, 'background thermal coeffiecient')
		
		ob_tp = inputbox_float(stdscr, 'object thermal coeffiecient')
		
		bk_temp = inputbox_int(stdscr, 'background temperature')
		
		ob_temp = inputbox_int(stdscr, 'object temperature')

		
	stdscr.clear()
	stdscr.addstr(0,0,'Object Simulation Mode\n',curses.A_BOLD)
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
	


	while True:
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
			
			heat_sys = heat_system(runtime, rate, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
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
			stdscr.addstr(0,0,'Plot must be closed to continue', curses.A_BOLD)
			stdscr.addstr(1,0,'Press <enter> to continue')
			stdscr.refresh()
			plt.show()
			stdscr.getch()
			plt.close('all')
			ani = anim_plot(gtot, heat_sys.grid)
			plt.show()
			contin(stdscr)
			plt.close('all')
			break
		else:
			stdscr.clear()
			stdscr.addstr('Press enter to reset...')
			stdscr.refresh()
			stdscr.getch()
			display3(stdscr)
			break
	
	stdscr.clear()
	if ret_main(stdscr):
		main(stdscr)
	else:
		stdscr.clear()
		stdscr.refresh()


#Display of Simulation mode 2
def display2(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to Step Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)

	cont = curses.newwin(3,40,5,10)
	cont.box()
	cont.addstr(1,1,'Continue')
	cont.refresh()
	stdscr.refresh()
	stdscr.getch()


	runtime = inputbox_int(stdscr, 'runtime')
	l = inputbox_int(stdscr, 'resolution')
	rate = inputbox_int(stdscr, 'framerate')
	top = inputbox_int(stdscr, 'top boundary temperature')
	right = inputbox_int(stdscr, 'right boundary temperature')
	bottom = inputbox_int(stdscr, 'bottom boundary temperature')
	left = inputbox_int(stdscr, 'left boundary temperature')
	bk_tp = inputbox_int(stdscr, 'background thermal coeffiecient')
	ob_tp = inputbox_int(stdscr, 'object thermal coeffiecient')
	bk_temp = inputbox_int(stdscr, 'background temperature')
	ob_temp = inputbox_int(stdscr, 'object temperature')

		
	stdscr.clear()
	stdscr.addstr(0,0,'Object Simulation Mode\n',curses.A_BOLD)
	stdscr.addstr(1,0,'Do these settings look right?\n',curses.A_BOLD)
	stdscr.addstr('Runtime : ' + str(runtime) + '\n')
	stdscr.addstr('Grid Length : ' + str(l) + '\n')
	stdscr.addstr('Framerate : ' + str(rate) + '\n')
	stdscr.addstr('Top : ' + str(top) + '\n')
	stdscr.addstr('Right : ' + str(right) + '\n')
	stdscr.addstr('Bottom : ' + str(bottom) + '\n')
	stdscr.addstr('Left : ' + str(left) + '\n')
	stdscr.addstr('Background thermal coeff. : ' + str(bk_tp) + '\n')
	stdscr.addstr('Object thermal coeff. : ' + str(ob_tp) + '\n')
	stdscr.addstr('Background temp : ' + str(bk_temp) + '\n')
	stdscr.addstr('Object temp : ' + str(ob_temp) + '\n')

	while True:
		curses.curs_set(1)
		inp = curses.newwin(3,10,5,20)
		inp.box()
		blurb = 'y/n?'
		inp.addstr(1,1,blurb)
		stdscr.refresh()
		curses.echo()
		a = len(blurb)+2
		conf = inp.getstr(1,a).decode('utf-8')
		curses.noecho()
		if conf =='Y' or conf =='y' :
			stdscr.clear()
			stdscr.addstr('Working...')
			stdscr.refresh()
			stdscr.getch()
			break
		else:
			stdscr.clear()
			stdscr.addstr('Press enter to reset...')
			stdscr.refresh()
			stdscr.getch()
			display1(stdscr)
			break

	stdscr.getch()


#Display of Simulation mode 3
def display3(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()

	stdscr.addstr(0,0,'Welcome to Sandbox Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)

	contin(stdscr)
	stdscr.clear()
	stdscr.refresh()
	op = 0
	op = menubox(stdscr, ['Default Settings', 'Customized'])
	
	if op ==1:
		runtime = 1000
		l = 100
		rate = 10
		top = 0
		right = 0
		bottom = 0
		left = 0
		bk_tp = 0.000001
		ob_tp = 0.000001
		bk_temp = 0
		ob_temp = 1
	else:
		runtime = inputbox_int(stdscr, 'runtime')
		l = inputbox_int(stdscr, 'resolution')
		rate = inputbox_int(stdscr, 'framerate')
		top = inputbox_int(stdscr, 'top boundary temperature')
		right = inputbox_int(stdscr, 'right boundary temperature')
		bottom = inputbox_int(stdscr, 'bottom boundary temperature')
		left = inputbox_int(stdscr, 'left boundary temperature')
		bk_tp = inputbox_float(stdscr, 'background thermal coeffiecient')
		ob_tp = inputbox_float(stdscr, 'object thermal coeffiecient')
		bk_temp = inputbox_int(stdscr, 'background temperature')
		ob_temp = inputbox_int(stdscr, 'object temperature')

		
	stdscr.clear()
	stdscr.addstr(0,0,'Object Simulation Mode\n',curses.A_BOLD)
	stdscr.addstr(1,0,'Do these settings look right?\n',curses.A_BOLD)
	
	s1='Runtime : ' + str(runtime)
	stdscr.addstr(2,0,s1)
	stdscr.addstr(2,50,'Recommended: ~1000')
	
	s2='Grid Length : ' + str(l)
	stdscr.addstr(3,0,s2)
	stdscr.addstr(3,50,'Recommended: ~100')
	
	s3='Framerate : ' + str(rate)
	stdscr.addstr(4,0,s3)
	stdscr.addstr(4,50,'Recommended: 5-10')
	
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
	


	while True:
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
			heat_sys = heat_system(runtime, rate, l, top, right, bottom, left, bk_tp, ob_tp, bk_temp, ob_temp)
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
			gtot = heat_jump(gtot, heat_sys.dt, heat_sys.dx, heat_sys.bk_tp, heat_sys.runtime, heat_sys.rate, heat_sys.ob_tp, heat_sys.is_ob)
			
			stdscr.clear()
			stdscr.addstr(0,0,'Creating plot and animation ...', curses.A_BOLD)
			stdscr.addstr(1,0,'Please wait', curses.A_BOLD)
			stdscr.refresh()
			doub_plot(gtot, heat_sys.x_size, heat_sys.grid)
			stdscr.clear()
			stdscr.addstr(0,0,'Plot must be closed to continue to animation', curses.A_BOLD)
			stdscr.refresh()
			plt.show()
			stdscr.getch()
			plt.close('all')
			ani = anim_plot(gtot, heat_sys.grid)
			plt.show()
			contin(stdscr)
			plt.close('all')
			break
		else:
			stdscr.clear()
			stdscr.addstr('Press enter to reset...')
			stdscr.refresh()
			stdscr.getch()
			display3(stdscr)
			break
	
	stdscr.clear()
	if ret_main(stdscr):
		main(stdscr)
	else:
		stdscr.clear()
		stdscr.refresh()
    
    
    
##########################################################################################
#End of Curses Functions


def main(stdscr):

    stdscr = curses.initscr()
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1,curses.COLOR_BLACK, curses.COLOR_GREEN)

    stdscr.clear()

    welcome= "Welcome to the Heat Equation Simulator! v1.1"
    stdscr.addstr(0,0,welcome,curses.A_BOLD)

    start_menu=["Simulation Mode","Step Mode","Sandbox Mode"]

    a = menubox(stdscr, start_menu)
    
    if a==1:
        display1(stdscr)
    elif a==2:
        display2(stdscr)
    elif a==3:
        display3(stdscr)
        
    stdscr.clear()
    stdscr.addstr(0,0, 'Thanks for using the Simulation!', curses.A_BOLD)
    curses.curs_set(1)
	
	# cleanup curses
    curses.endwin()

    stdscr.getch()
		
	



curses.wrapper(main)


