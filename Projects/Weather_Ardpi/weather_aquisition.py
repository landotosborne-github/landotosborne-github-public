#!/usr/bin/env python3

import serial
import numpy as np
from time import sleep

ser = serial.Serial('/dev/ttyACM0',9600)
filename = 'weather_stats.csv'

data = np.loadtxt(filename,delimiter=',', skiprows=1)

sample = np.empty((0,3),dtype=float)
for i in range(10):
	inb = ser.readline()
	inlist = np.array(inb.decode('utf-8').strip().replace(',',' ').split(),dtype=float)
	sample = np.append(sample,[inlist],axis=0)
	sleep(1)


sample = np.mean(sample,axis=0)


data = np.append(data, [sample], axis=0)

size = data.shape[0]

if size > 144:
	data = data[(size-144):,:]

np.savetxt(filename, data, delimiter=',', header='Temp,Humidity,Lux', fmt='%.1f')

