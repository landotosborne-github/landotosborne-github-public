"""
Created on Mon Oct 24 20:17:49 2022

@author: landonosborne
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.constants import k as boltk


def space():
    return print("")


f = open("AMP1.TXT","r")
g = open("AMP2.TXT","r")
h = open("AMP3.TXT","r")

i = open("BG1.TXT","r")
j = open("BG2.TXT","r")
k = open("BG3.TXT","r")

l = open("jonsonnoise - ROOM TEMP.csv","r")
m = open("jonsonnoise - L_N2.csv","r")


datf1=pd.read_csv(f, header=None)
datf2=pd.read_csv(g, header=None)
datf3=pd.read_csv(h, header=None)

bgf1=pd.read_csv(i, header=None)
bgf2=pd.read_csv(j, header=None)
bgf3=pd.read_csv(k, header=None)

rTf=pd.read_csv(l,header=None)
LN2f=pd.read_csv(m,header=None)




data1 = datf1[[0,1]].to_numpy()
data2 = datf2[[0,1]].to_numpy()
data3 = datf3[[0,1]].to_numpy()

bkg1 = bgf1[[0,1]].to_numpy()
bkg2 = bgf2[[0,1]].to_numpy()
bkg3 = bgf3[[0,1]].to_numpy()



rT=rTf[[0,1]].to_numpy()
LN2=LN2f[[0,1]].to_numpy()


freq=data1[:,0]

d1d=data1[:,1]
d2d=data2[:,1]
d3d=data3[:,1]

vrmsAvDat = ((d1d+d2d+d3d)/3)



bg1d=bkg1[:,1]
bg2d=bkg2[:,1]
bg3d=bkg3[:,1]

vrmsAvBkg = ((bg1d+bg2d+bg3d)/3)

dataTot=np.stack((freq,vrmsAvDat),axis=1)
bkgTot=np.stack((freq,vrmsAvBkg),axis=1)




gain=np.divide(vrmsAvDat,vrmsAvBkg)
gainFcn=np.stack((freq,gain),axis=1)

dataTotF=pd.DataFrame(dataTot,columns=["Frequency (Hz)","PreAmp"])
bkgTotF=pd.DataFrame(bkgTot,columns=["Frequency (Hz)","Background"])
gainFcnF=pd.DataFrame(gainFcn,columns=["Frequency (Hz)","Gain"])


f=bkgTotF.plot(x="Frequency (Hz)",y="Background", color = "blue")
dataTotF.plot(ax=f,x="Frequency (Hz)",y="PreAmp",color="red")
plt.title("Background vs. PreAmp Voltage")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Vrms (V)")
plt.show()




def Gauss(x, a, mu, sig):
    y = a*np.exp(-(x-mu)**2/(2*(sig**2)))
    return y


fitx = gainFcn[:,0]
fity = gainFcn[:,1]
a = max(fity)
z=0


mu=np.sum(fitx*fity)/np.sum(fity)
sig= np.sqrt(np.sum(fity*(fitx-mu)**2)/np.sum(fity))



paramG,covG = curve_fit(Gauss, fitx, fity,p0=[a,mu,sig])

a = paramG[0]
mu = paramG[1]
sig = paramG[2]

err_G = np.sqrt(np.diag(covG))

err_a= err_G[0]
err_mu= err_G[1]
err_sig= err_G[2]


err_C = 0.05*10**-12

err_R = np.array([1.200000E+01,1.300000E+02,1.800000E+02,
         2.600000E+02,3.200000E+02,1.300000E+03])

err_V2_rT= np.array([ 2.022373E-08,1.953710E-07,3.886835E-07
         ,6.715918E-07,1.433494E-06,1.770580E-06])

err_V2_LN2=np.array([5.518592E-09,5.449872E-08,1.070043E-07,
                     1.840670E-07,2.428340E-07,4.243799E-07])




fitted_gaussian=Gauss(fitx,a,mu,sig)

plt.plot(freq,gain,"green",label="Gain")
plt.plot(fitx,fitted_gaussian,"blue",label="Fit")
plt.title("Gain function with Gaussian Fit")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain")
plt.legend()
plt.show()


print("a  = ", a,"mu = ", mu ,"sig = ", sig )
print("Gaussian uncertainties are " , err_G)
space()


def Integrand(x,a,mu,sig,r):
    y = a*np.exp(-(x-mu)**2/(2*sig**2))
    value = y**2/(1+(2*np.pi*x*(6.01*10**-12)*r)**2)
    return value

def partA(x,a,mu,sig,r):
    y = 2*a*np.exp(-(x-mu)**2/(sig**2))
    value = y/(1+(2*np.pi*x*(6.01*10**-12)*r)**2)
    return value

def partmu(x,a,mu,sig,r):
    y = 2*a**2*(x-mu)*np.exp(-(x-mu)**2/(sig**2))
    value = y/(sig**2*(1+(2*np.pi*x*(6.01*10**-12)*r)**2))
    return value

def partsig(x,a,mu,sig,r):
    y = 2*a**2*(x-mu)**2*np.exp(-(x-mu)**2/(sig**2))
    value = y/(sig**3*(1+(2*np.pi*x*(6.01*10**-12)*r)**2))
    return value

def partC(x,a,mu,sig,r):
    y = -8*np.pi**2*6.01*10**-12*r**2*x**2*a**2*np.exp(-(x-mu)**2/(sig**2))
    value = y/((1+(2*np.pi*x*(6.01*10**-12)*r)**2))**2
    return value

def partR(x,a,mu,sig,r):
    y = -8*np.pi**2*(6.01*10**-12)**2*r*x**2*a**2*(x-mu)*np.exp(-(x-mu)**2/(sig**2))
    value = y/((1+(2*np.pi*x*(6.01*10**-12)*r)**2))**2
    return value



R=LN2[:,0]
Band_array=[]
Band_err=[]
for i in range(len(R)):
    Band_Gain=quad(Integrand,3000,10000,args=(a,mu,sig,R[i]))
   
    eA= quad(partA,3000,10000,args=(a,mu,sig,R[i]))
    eA= (eA[0]*err_a)**2
    
    eMu=quad(partmu,3000,10000,args=(a,mu,sig,R[i]))
    eMu= (eMu[0]*err_mu)**2
    
    eSig=quad(partsig,3000,10000,args=(a,mu,sig,R[i]))
    eSig= (eSig[0]*err_sig)**2
    
    eC=quad(partC,3000,10000,args=(a,mu,sig,R[i]))
    eC= (eC[0]*err_C)**2
    
    eR=quad(partR,3000,10000,args=(a,mu,sig,R[i]))
    eR= (eR[0]*err_R[i])**2
    
    errBG= np.sqrt(eA+eMu+eSig+eC+eR)
    
    Band_array.append(Band_Gain[0])
    Band_err.append(errBG)


print('My Band Gain values are: ',Band_array)
print("The Band Gain error is: ", Band_err)
space()


y_temp_rT=rT[:,1]
y_temp_LN2=LN2[:,1]
y_k_rT=rT[:,1]
y_k_LN2=LN2[:,1]


def Linear(x,a):
    y=a*x
    return y




y_temp_rT=y_temp_rT/(2*boltk)
y_temp_rT= np.divide(y_temp_rT, Band_array)

y_temp_rT_err =[]
for i in range(len(R)):
    err= y_temp_rT[i]*np.sqrt((err_V2_rT[i]/rT[i,1])**2 +(Band_err[i]/Band_array[i])**2)
    y_temp_rT_err.append(err)


paramR,covR = curve_fit(Linear, R, y_temp_rT,p0=[298],sigma=y_temp_rT_err,absolute_sigma=True)

plt.plot(R,y_temp_rT,"o")

tR=paramR[0]

fitrT=Linear(R,tR)
plt.plot(R,fitrT,"blue")
plt.errorbar(R,y_temp_rT,yerr=y_temp_rT_err,ecolor="darkorange",fmt="o")
plt.xlabel("Resistance  (\u03A9)")
plt.ylabel("Noise Voltage")
plt.title("Linear fit of Noise Voltage at room temperature")
plt.show()

print("Calculated room temperature is ",tR)
print("Uncertainty is ", np.sqrt(np.diag(covR)))
print("Actual room temperature is 295K")
space()




y_temp_LN2=y_temp_LN2/(2*boltk)
y_temp_LN2= np.divide(y_temp_LN2, Band_array)

y_temp_LN2_err=[]
for i in range(len(R)):
    err= y_temp_LN2[i]*np.sqrt((err_V2_LN2[i]/LN2[i,1])**2 +(Band_err[i]/Band_array[i])**2)
    y_temp_LN2_err.append(err)


paramN,covN = curve_fit(Linear,R, y_temp_LN2, p0=[77], sigma=y_temp_LN2_err,absolute_sigma=True)

tN=paramN[0]

plt.plot(R,y_temp_LN2,"o")

fitLN2= Linear(R,tN)
plt.plot(R,fitLN2)
plt.errorbar(R,y_temp_LN2,yerr=y_temp_LN2_err,ecolor="green",fmt="o")
plt.xlabel("Resistance  (\u03A9)")
plt.ylabel("Noise Voltage (V)")
plt.title("Linear fit of Noise Voltage in liquid Nitrogen")
plt.show()

print("Calculated temperature of liquid nitrogen is ",tN)
print("Uncertainty is ", np.sqrt(np.diag(covN)))
print("Actual temperature is 77K")
space()




RG=Band_array

y_k_rT=y_k_rT/2
y_k_rT=np.divide(y_k_rT,RG)

y_k_rT_err=[]
for i in range(len(R)):
    err= y_k_rT[i]*np.sqrt((err_V2_rT[i]/rT[i,1])**2 +(Band_err[i]/Band_array[i])**2+(err_R[i]/R[i])**2)
    y_k_rT_err.append(err)


param_kb_rT,cov_kb_rT=curve_fit(Linear,R*298,y_k_rT,sigma=y_k_rT_err,absolute_sigma=True)

print('The calculated value with my measurements for the boltzmann constant @ room temp is',param_kb_rT[0])
print('Uncertainty is',np.sqrt(np.diag(cov_kb_rT)))
space()



y_k_LN2=y_k_LN2/2
y_k_LN2=np.divide(y_k_LN2,RG)

y_k_LN2_err=[]
for i in range(len(R)):
    err= y_k_LN2[i]*np.sqrt((err_V2_LN2[i]/LN2[i,1])**2 +(Band_err[i]/Band_array[i])**2+(err_R[i]/R[i])**2)
    y_k_LN2_err.append(err)



param_kb_LN2,cov_kb_LN2=curve_fit(Linear,R*77,y_k_LN2,sigma=y_k_LN2_err,absolute_sigma=True)

print('The calculated value with my measurements for the boltzmann constant w/ LN2 is',param_kb_LN2[0])
print('Uncertainty is',np.sqrt(np.diag(cov_kb_LN2)))
space()
print("Actual value of boltzman constant is ", boltk)
space()
print(param_kb_LN2[0]/boltk)







