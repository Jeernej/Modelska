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

e=2.718281828459045    
pi=3.141592653589793 

barva=['r','b','g','k','y']


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/dvanajsta/'
val = ["val2","val3"]

SIG = loadtxt(DIR + val[0]+".dat") # branje
SIG2 = loadtxt(DIR + val[1]+".dat") # branje
LS= len(SIG) # dolžina  signala

SIG_256= SIG[0:LS//2]
SIG_128= SIG[0:LS//4]
SIG_64= SIG[0:LS//6]

SIG2_256= SIG2[0:LS//2]
SIG2_128= SIG2[0:LS//4]
SIG2_64= SIG2[0:LS//6]

FTSIG = 2*abs(np.fft.fft(SIG))**2 
FTfreq =np.fft.fftfreq(LS, 1/len(SIG))
FTSIG_256 = 2*abs(np.fft.fft(SIG_256))**2 
FTfreq_256 =np.fft.fftfreq(LS//2, 1/len(SIG_256))
FTSIG_128 = 2*abs(np.fft.fft(SIG_128))**2  
FTfreq_128 =np.fft.fftfreq(LS//4, 1/len(SIG_128))
FTSIG_64 = 2*abs(np.fft.fft(SIG_64))**2  
FTfreq_64 =np.fft.fftfreq(LS//6, 1/len(SIG_64))

FTSIG2 = 2*abs(np.fft.fft(SIG2))**2  
FTfreq2 =np.fft.fftfreq(LS, 1/len(SIG2))
FTSIG2_256 = 2*abs(np.fft.fft(SIG2_256))**2  
FTfreq2_256 =np.fft.fftfreq(LS//2, 1/len(SIG2_256))
FTSIG2_128 = 2*abs(np.fft.fft(SIG2_128))**2  
FTfreq2_128 =np.fft.fftfreq(LS//4, 1/len(SIG2_128))
FTSIG2_64 = 2*abs(np.fft.fft(SIG2_64))**2  
FTfreq2_64 =np.fft.fftfreq(LS//6, 1/len(SIG2_64))

#SIG1 = signal.resample(SIG, LS2)   #
#SIG2 = signal.resample(SIG2, LS1)
#LS1= len(SIG1) # dolžina  signala
#LS2= len(SIG2)
#KOR=signal.fftconvolve(SIG1, SIG2[::-1], mode='same')
#FT1 = fft(SIG1) 

T=np.linspace(0,len(SIG),len(SIG)) /len(SIG)
freq=np.linspace(0,len(FTSIG),len(FTSIG))#-len(FTSIG)/2


F50=plt.figure(50)
F50=plt.subplot(3, 1, 1 ) 
plt.title('Izmerjena signala '+val[0]+'.dat in '+val[1]+'.dat ',fontsize=16)
plt.plot(T,SIG,color='c',ls='-',alpha = 0.5, label=val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,SIG2,color='m',ls='-',alpha = 0.5, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[0:LS//2],SIG_256,color=barva[1],ls=':',alpha = 0.5, label='N='+str(LS//2)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[0:LS//4],SIG_128,color=barva[2],ls=':',alpha = 0.5, label='N='+str(LS//4)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[0:LS//6],SIG_64,color=barva[3],ls=':',alpha = 0.5, label='N='+str(LS//6)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
#plt.yscale('log')
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
#    plt.xlim([0,6])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=1)

F50=plt.subplot(3, 1, 2 ) 
plt.title('FFT izmerjenega signala '+val[0]+'.dat  za različne intervale',fontsize=16)
plt.plot(freq,FTSIG,color=barva[3],ls='-',alpha = 0.5,label='N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//2]*2,FTSIG_256,color=barva[1],ls='-',alpha = 0.5,label='N='+str(LS//2))#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//4]*4,FTSIG_128,color=barva[2],ls='-',alpha = 0.5,label='N='+str(LS//4))#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//6]*6,FTSIG_64,color=barva[0],ls='-',alpha = 0.5,label='N='+str(LS//6))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

F50=plt.subplot(3, 1, 3 ) 
plt.title('FFT izmerjenega signala '+val[1]+'.dat za različne intervale',fontsize=16)
#plt.plot(FTfreq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG2,color=barva[3],ls='-',alpha = 0.5,label='N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//2]*2,FTSIG2_256,color=barva[1],ls='-',alpha = 0.5,label='N='+str(LS//2))#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//4]*4,FTSIG2_128,color=barva[2],ls='-',alpha = 0.5,label='N='+str(LS//4))#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//6]*6,FTSIG2_64,color=barva[0],ls='-',alpha = 0.5,label='N='+str(LS//6))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,len(SIG)/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

###############################################################################
################  okenske funkcije in odzivi #################################
###############################################################################

windowGauss = signal.gaussian(51, std=6)
windowBartt = signal.bartlett(51)
windowHann = signal.hann(51)
windowCH = signal.chebwin(51, at=100)
windowTuR = signal.tukey(51)

AGauss = np.fft.fft(windowGauss, 2048) / (len(windowGauss)/2.0)
ABartt = np.fft.fft(windowBartt, 2048) / (len(windowBartt)/2.0)
AHann = np.fft.fft(windowHann, 2048) / (len(windowHann)/2.0)
ACH = np.fft.fft(windowCH, 2048) / (len(windowCH)/2.0)
ATuR = np.fft.fft(windowTuR, 2048) / (len(windowTuR)/2.0)

freqGauss = np.linspace(-0.5, 0.5, len(AGauss))
freqBartt = np.linspace(-0.5, 0.5, len(ABartt))
freqHann = np.linspace(-0.5, 0.5, len(AHann))
freqCH = np.linspace(-0.5, 0.5, len(ACH))
freqTuR = np.linspace(-0.5, 0.5, len(ATuR))

responseGauss  = 20 * np.log10(np.abs(fftshift(AGauss / abs(AGauss).max())))
responseBartt = 20 * np.log10(np.abs(fftshift(ABartt / abs(ABartt).max())))
responseHann = np.abs(fftshift(AHann / abs(AHann).max()))
responseHann = 20 * np.log10(np.maximum(responseHann, 1e-10))
responseCH = 20 * np.log10(np.abs(fftshift(ACH / abs(ACH).max())))
responseTuR = 20 * np.log10(np.abs(fftshift(ATuR  / abs(ATuR ).max())))

#FT2 = np.ndarray.conjugate(fft(SIG2))
#KOR=ifft(FT1*(FT2))  

F1=plt.figure(1)
F1=plt.subplot(1, 2, 1 ) 
plt.plot(windowGauss, barva[0], label=r'Gauss ($\sigma$=6)')
plt.plot(windowBartt, barva[1], label='Bartlett')
plt.plot(windowHann, barva[2], label='Hann')
plt.plot(windowCH, barva[3], label='Dolph-Chebyshev (100 dB)')
plt.plot(windowTuR, barva[4], label='Turkey')
plt.title("Okenske funkcije",fontsize=16)
plt.ylabel("Amplituda",fontsize=16)
plt.xlabel("Signal",fontsize=16)
plt.legend(loc=0)


F1=plt.subplot(1, 2, 2 ) 
plt.plot(freqGauss, responseGauss, barva[0], alpha = 0.5, label=r'Gauss ($\sigma$=6)')
plt.plot(freqBartt, responseBartt, barva[1], alpha = 0.5, label='Bartlett')
plt.plot(freqHann, responseHann, barva[2], alpha = 0.5, label='Hann')
plt.plot(freqCH, responseCH, barva[3], alpha = 0.5, label='Dolph-Chebyshev (100 dB)')
plt.plot(freqTuR, responseTuR, barva[4], alpha = 0.95, label='Turkey')
plt.axis([-0.5, 0.5, -120, 0])
plt.title("Frekvenčni odziv na okenske funkcije",fontsize=16)
plt.ylabel("Normalizirana magnituda I [dB]",fontsize=16)
plt.xlabel(r'Normalizirana frekvenca $\omega$ [cikel/signal]',fontsize=16)
plt.legend(loc=0)

###############################################################################
###############################################################################
###############################################################################




#
#for j in range(0,3):
#    
#    L0=50.
#    Z0=200.
#    t_smrti=0    
#
#    Lt=np.empty(0)
#    Zt=np.empty(0)
#    T=np.empty(0)
#
#    Lt=np.append(Lt, L0)
#    Zt=np.append(Zt, Z0)
#    T=np.append(T, t_smrti)
#
##    N0=25
#
#    dt=dT[j]
#    
#    i=0
#    while  Zt[i]>0 and Lt[i]>0:
#        
#        dZ1=np.random.poisson(lam=5.0*Zt[i]*dt, size=None) #rojstvo
#        dL1=np.random.poisson(lam=4.0*Lt[i]*dt, size=None)         #rojstvo
#        dZ2=np.random.poisson(lam=4.0*Zt[i]*dt, size=None)  #smrt
#        dL2=np.random.poisson(lam=5.0*Lt[i]*dt, size=None)        #smrt
#        dZ3=np.random.poisson(lam=0.02*Zt[i]*Lt[i]*dt, size=None) #smrt
#        dL3=np.random.poisson(lam=0.005*Zt[i]*Lt[i]*dt, size=None) #rojstvo
#        
#        Zt=np.append(Zt, Zt[i]+dZ1-dZ2-dZ3)
#        Lt=np.append(Lt, Lt[i]+dL1-dL2+dL3)
#        
##        dZ1=np.random.poisson(lam=5.0*Zt[i]*dt, size=None) #rojstvo
##        dL1=np.random.poisson(lam=4.0*Lt[i]*dt + 0.005*Zt[i]*Lt[i]*dt, size=None)         #rojstvo
##        dZ2=np.random.poisson(lam=4.0*Zt[i]*dt + 0.02*Zt[i]*Lt[i]*dt, size=None)  #smrt
##        dL2=np.random.poisson(lam=5.0*Lt[i]*dt, size=None)        #smrt
##        
##        Zt=np.append(Zt, Zt[i]+dZ1-dZ2)
##        Lt=np.append(Lt, Lt[i]+dL1-dL2)     
#        
#        
#        t_smrti=t_smrti+dt
#        T=np.append(T, t_smrti)
#        
#        
#        i=i+1
#        
#    
#    F50=plt.figure(50)
#    plt.suptitle(r'Umiranje populacije računano z različnimi časovnimi koraki',fontsize=16)
#    F50=plt.subplot(1, 2, 1 ) 
#    plt.plot(T,Lt,color=crta[j],ls=':',alpha = 0.95,label=r'lisice $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Zt,color=crta[j],ls='--',alpha = 0.95,label=r'zajci $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
##    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
##    plt.xscale('log')
##    plt.yscale('log')
#    plt.ylabel(' N(t) ')   
#    plt.xlabel(' t ')
##    plt.xlim([0,6])
##    plt.ylim([0,250])
##    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#    plt.legend(loc=0)
#    
#    F50=plt.subplot(1, 2, 2 )  
#    plt.plot(Lt,Zt,color=crta[j],alpha = 0.95,label=r'$\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
##    plt.xscale('log')
##    plt.yscale('log')
#    plt.ylabel(' Z(t) ')
#    plt.xlabel(' L(t) ')
##    plt.xlim([0,6])
##    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#    plt.legend(loc=0)
#
