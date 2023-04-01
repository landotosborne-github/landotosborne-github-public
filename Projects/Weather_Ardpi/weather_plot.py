#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

filename = 'weather_stats.csv'
data = np.loadtxt(filename,delimiter=',', skiprows=1)

end = dt.datetime.now()
formatted_date = end.strftime('%H:%M:%S %Y-%m-%d')


start = end - dt.timedelta(hours=24)


trange = [start + dt.timedelta(minutes=10*i) for i in range(-143,1)]

f1 = plt.figure(figsize=(15,10))

ax1 = f1.add_subplot(221)
ax1.plot(trange, data[:,0] ,color='red')
ax1.set_title('Temperature (F)')
ax1.set_ylim(45,90)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

curr_temp = data[-1,0]

ax2 = f1.add_subplot(222)
ax2.plot(trange,data[:,1], color='blue')
ax2.set_title('Humidity (%)')
ax2.set_ylim(-1,101)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

curr_hum = data[-1,1]

ax3 = f1.add_subplot(212)
ax3.plot(trange,data[:,2], color='green')
ax3.set_title('Lux')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

curr_lux = data[-1,2]

f1.suptitle(f'{formatted_date}  |  Temp (F): {curr_temp} , H (%): {curr_hum} , Lux : {curr_lux}')
f1.subplots_adjust(wspace=0.3,hspace=0.5)
f1.savefig('current_weather.pdf',format='pdf')
