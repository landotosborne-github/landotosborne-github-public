# landotosborne-github-public/Projects/Weather_Ardpi/weather_aquisition

##WORK IN PROGRESS##

This project is something I have been working on for just a short while.Its goal is to create a remote fully automated weather station that records weather conditions with sensors. This data is recieved from the computer with serial communication and added to a consistently updating csv file. At any point the current conditions and conditions from the past 24 hours can be plotted into a neat graphical pdf.

The hardware I used was a Raspberry Pi 400, and an Arduino Uno for serial data acquisition. By adding weather_aquisition.py and weather_plot.py to the crontab these jobs can be updated regularly and are available to view at any time via a ssh connection.
