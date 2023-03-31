#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from numba import jit
import curses


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
		self.dt = dt
		self.x_size = x_size
		self.dx = self.x_size / self.grid

#initialize pixels for simulation
	def init_sys(self):
		frames = self.runtime // self.rate
		gtot = np.zeros((frames , self.grid , self.grid))
		
		#set background temp
		gtot[0] = self.bk_temp
		
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


#Calculate next frame of simulation
@jit
def heat_step(g, dt , dx , tp):
	ng = g.copy()
	alph = (tp*dt)/dx**2
	for i in range(1 ,l-1):
		for j in range(1,l-1):
			ng[j][i] = g[j][i] + (alph)*(g[j+1][i] + g[j-1][i] + g[j][i+1]+g[j][i-1] - 4*g[i][j])
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
	a = ax2.imshow(gtot[frames-1] , cmap='inferno', extent=[0,l,0,l])
	f1.colorbar(a, ax=ax2)
	f1.show()


# Create colormap animation of system
def anim_plot(gtot):
	f = plt.figure(figsize = (5,5))
	ax = f.add_subplot(111)
	a = ax.imshow(gtot[0] , cmap='inferno', extent=[0,1,0,1])
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
#Curses Functions

#Creates a input box for user to input integer values
def inputbox_int(stdscr,val):
	while True:
		stdscr.clear()
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
			break
		except ValueError:
			curses.curs_set(0)
			inp.addstr(1,1,'Incorrect Data type. Please enter integer.')
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


#Display of Simulation mode 1 
def display1(stdscr):
	stdscr.clear()
	stdscr.refresh()
	curses.curs_set(0)
	curses.start_color()
	stdscr.addstr(0,0,'Welcome to Simulation Mode\nPlease choose the settings for your simulation ...\n',curses.A_BOLD)
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
		inp = curses.newwin(3,10,5,40)
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
			curses.curs_set(0)
			stdscr.addstr(0,0,'How would you like the internal temperature to be set?')
			b = menubox(stdscr,['Random','Abs(Random)','Centered Dot'])
			if b == 1:
				pass
			elif b == 2:
				pass
			elif b == 3:
				pass
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
    
    
    
##########################################################################################
#End of Curses Functions


def main(stdscr):

    stdscr = curses.initscr()
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1,curses.COLOR_BLACK, curses.COLOR_GREEN)

    stdscr.clear

    welcome= "Welcome to the Heat Equation Simulator! v1.0"
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


