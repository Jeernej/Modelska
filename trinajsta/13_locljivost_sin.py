# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
#import scipy as sc
import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D
#from scipy import stats
#from scipy.optimize import curve_fit
import numpy as np
import scipy.fftpack
from numpy import loadtxt
from numpy import savetxt
#from numpy import fft
#from scipy.special import airy, gamma
#from scipy import fftconvolve
#from scipy import convolve
from scipy import signal
from scipy.fftpack import fftfreq ,fftshift, fft
from scipy import linalg
from scipy.linalg import solve
from scipy.linalg import solve_toeplitz
from scipy.optimize import root

e=2.718281828459045    
pi=3.141592653589793 

barva=['r','b','g','y','m','c','o']

LS= 512 # dolžina  signala
#SIG = [3*np.sin(5*2*np.pi*k/10)+2*np.sin(10*2*np.pi*k/20)+1*np.sin(15*2*np.pi*k/30) for k in range(LS)]
SIG=[100*np.sin(60*2*np.pi*k/LS)+50*np.sin(70*2*np.pi*k/LS)+50*np.sin(100*2*np.pi*k/LS)+100*np.sin(105*2*np.pi*k/LS) for k in range(LS)]

T=np.linspace(0,len(SIG),len(SIG)) /len(SIG)


########### FOURIERE ############################

FTSIG = 2*abs(np.fft.fft(SIG))**2 
FTfreq =np.fft.fftfreq(LS, 1/len(SIG))

FTSIG_H = 2*abs(np.fft.fft(SIG*signal.hann(LS)))**2 
FTSIG_CH = 2*abs(np.fft.fft(SIG*signal.chebwin(LS, at=100)))**2 

freq=np.linspace(0,len(FTSIG),len(FTSIG))#-len(FTSIG)/2


########### MAX entrophy ############################

def R_ak(i,S,N): ## avtokorelacijske funkcije R
    return sum( S[n]*S[n+i] / (i-N) for n in range((N-1-i)-1))
    
def M(S,Npoli,N): ## matrika avtokorelacijskih funkcij R
    return np.array([[R_ak(abs(i-j),S,N) for i in range(Npoli)] for j in range(Npoli)])
    
def b(S,Npoli,N): ## vektor avtokorelacijskih funkcij -R
    return np.array([-R_ak(i+1,S,N) for i in range(Npoli)])

def a(S,Npoli):  # koeficienti rešitev 
    N=len(S)
    return np.linalg.solve(M(S,Npoli,N), b(S,Npoli,N)) ### solving the Yule-Walker Equations

#def a_h(S,p):
#    N=len(S)
#    c=[R_ak(i,S,N) for i in range(p)]
#    b=[-R_ak(i+1,S,N) for i in range(p)]
#    return linalg.solve_toeplitz(c, b)  ### solving the Yule-Walker Equations with Toeplitz system

def z(ak):
    Npoli=len(ak)
    p=[ak[Npoli-1-i] if i<Npoli else 1 for i in range(Npoli+1)]
    return (np.roots(p))**-1  # kompleksne Z rešitve za   p[0] * Z**n + p[1] * Z**(n-1) + ... + p[n-1]*Z + p[n]=

def z_in(zk):
    Z_in=(zk/abs(zk))**-1
    return Z_in  # kompleksne Z rešitve za   p[0] * Z**n + p[1] * Z**(n-1) + ... + p[n-1]*Z + p[n]=


def ak_abs(zk,ak):
    Npoli=len(ak)
    z=z_in(zk)
    for i in range(len(z)) : z[i]=1/z[i]
    rešitev=np.poly1d(z, True)                           ### iskanje novih ak-jev z normiranimi Z-ji
    return [rešitev[i+1]/rešitev[0] for i in range(Npoli)]

def P(ni,ak,N): ##### računanje spektra PSD
    Npoli=len(ak)
    return 1/(abs(1+sum(ak[i]*np.exp(-1j*ni*2*np.pi*(i+1)/(N/2)) for i in range(Npoli)))**2)  



ali2=a(SIG,5)
zk2=z(ali2)
alin2=ak_abs(zk2,ali2)


##################################################################################
F60=plt.figure(60)
F60=plt.subplot(1, 1, 1 ) 
plt.title('Poli za vzorec  ;  N='+str(LS),fontsize=16)
c=0
for i in [6,8,10,12]: # števila polov
    ali3=a(SIG,i)
    zk3=z(ali3)
    zk3_in=z_in(zk3)
#    plt.scatter(zk3.real,zk3.imag, color=barva[c],alpha = 0.95, marker='x', label='p={}'.format(i))
    plt.scatter(zk3_in.real,zk3_in.imag, color=barva[c],alpha = 0.95,  marker='p',label='p={} ; (|z|<1)'.format(i))
    c=c+1

circle1 = plt.Circle((0, 0), 1, color=barva[0],fill=False)
F60.add_artist(circle1)

plt.xlabel('Re[z]',fontsize=16)
plt.ylabel('Im[z]',fontsize=16)
plt.legend(loc='best')




# val2
F20=plt.figure(20)
F20=plt.subplot(2, 1, 1 ) 
plt.title('Generiran sinusni vzorec ',fontsize=16)
plt.plot(T,SIG,color='c',ls='-',alpha = 0.85)#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.legend(loc=1)

F20=plt.subplot(2, 1, 2 ) 
plt.title('Spekter sinusnega vzorca ;  N='+str(LS),fontsize=16)
c=0
for i in [8,12,14,18]: # števila polov
    ali2=a(SIG,i)
    zk2=z(ali2)
    alin2=ak_abs(zk2,ali2)
#    plt.plot(freq*2,[P(i,ali2,LS) for i in range(LS)],color=barva[c],ls='--',label='p={}'.format(i))
    plt.plot(freq*2,[P(i,alin2,LS) for i in range(LS)],color=barva[c],label='p={} ; (|z|<1)'.format(i))
    c=c+1
plt.plot(freq,FTSIG,color='k',ls='-',alpha = 0.65,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG_H,color=barva[2],ls='-',alpha = 0.5,label='FFT + Hann  ')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG_CH,color='k',ls='--',alpha = 0.65,label='FFT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS/2])
plt.ylim([0.00000001,10000000000])
plt.legend(loc=1)
plt.yscale('log')


