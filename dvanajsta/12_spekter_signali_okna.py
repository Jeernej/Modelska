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

barva=['r','b','g','k','m','y','c']


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/dvanajsta/'
val = ["val2","val3"]
V=1
SIG = loadtxt(DIR + val[V]+".dat") # branje
LS= len(SIG) # dolžina  signala

SIG_256= SIG[0:LS//2]
SIG_128= SIG[0:LS//4]
SIG_64= SIG[0:LS//8]

STD=7
windowGauss = signal.gaussian(LS, std=STD)
windowBartt = signal.bartlett(LS)
windowHann = signal.hann(LS)
windowCH = signal.chebwin(LS, at=100)
windowTuR = signal.tukey(LS)

FTSIG = 2*abs(np.fft.fft(SIG))**2 
FTfreq =np.fft.fftfreq(LS, 1/len(SIG))
FTSIG_G = 2*abs(np.fft.fft(SIG*signal.gaussian(LS, std=STD)))**2 
FTSIG_B = 2*abs(np.fft.fft(SIG*signal.bartlett(LS)))**2 
FTSIG_H = 2*abs(np.fft.fft(SIG*signal.hann(LS)))**2 
FTSIG_CH = 2*abs(np.fft.fft(SIG*signal.chebwin(LS, at=100)))**2 
FTSIG_T= 2*abs(np.fft.fft(SIG*signal.tukey(LS)))**2 
FTSIG_HCH= 2*abs(np.fft.fft(SIG*signal.tukey(LS)*signal.hann(LS)))**2 


FTSIG_256 = 2*abs(np.fft.fft(SIG_256))**2 
FTfreq_256 =np.fft.fftfreq(LS//2, 1/len(SIG_256))
FTSIG_G_256 = 2*abs(np.fft.fft(SIG_256*signal.gaussian(LS//2, std=STD)))**2 
FTSIG_B_256 = 2*abs(np.fft.fft(SIG_256*signal.bartlett(LS//2)))**2 
FTSIG_H_256 = 2*abs(np.fft.fft(SIG_256*signal.hann(LS//2)))**2 
FTSIG_CH_256 = 2*abs(np.fft.fft(SIG_256*signal.chebwin(LS//2, at=100)))**2 
FTSIG_T_256= 2*abs(np.fft.fft(SIG_256*signal.tukey(LS//2)))**2 
FTSIG_HCH_256= 2*abs(np.fft.fft(SIG_256*signal.tukey(LS//2)*signal.hann(LS//2)))**2 

FTSIG_128 = 2*abs(np.fft.fft(SIG_128))**2  
FTfreq_128 =np.fft.fftfreq(LS//4, 1/len(SIG_128))
FTSIG_G_128 = 2*abs(np.fft.fft(SIG_128*signal.gaussian(LS//4, std=STD)))**2 
FTSIG_B_128 = 2*abs(np.fft.fft(SIG_128*signal.bartlett(LS//4)))**2 
FTSIG_H_128 = 2*abs(np.fft.fft(SIG_128*signal.hann(LS//4)))**2 
FTSIG_CH_128 = 2*abs(np.fft.fft(SIG_128*signal.chebwin(LS//4, at=100)))**2 
FTSIG_T_128= 2*abs(np.fft.fft(SIG_128*signal.tukey(LS//4)))**2 
FTSIG_HCH_128= 2*abs(np.fft.fft(SIG_128*signal.tukey(LS//4)*signal.hann(LS//4)))**2 

FTSIG_64 = 2*abs(np.fft.fft(SIG_64))**2  
FTfreq_64 =np.fft.fftfreq(LS//8, 1/len(SIG_64))
FTSIG_G_64 = 2*abs(np.fft.fft(SIG_64*signal.gaussian(LS//8, std=STD)))**2 
FTSIG_B_64 = 2*abs(np.fft.fft(SIG_64*signal.bartlett(LS//8)))**2 
FTSIG_H_64 = 2*abs(np.fft.fft(SIG_64*signal.hann(LS//8)))**2 
FTSIG_CH_64 = 2*abs(np.fft.fft(SIG_64*signal.chebwin(LS//8, at=100)))**2 
FTSIG_T_64= 2*abs(np.fft.fft(SIG_64*signal.tukey(LS//8)))**2  
FTSIG_HCH_64= 2*abs(np.fft.fft(SIG_64*signal.tukey(LS//8)*signal.hann(LS//8)))**2 

#SIG1 = signal.resample(SIG, LS2)   #
#SIG2 = signal.resample(SIG2, LS1)
#LS1= len(SIG1) # dolžina  signala
#LS2= len(SIG2)
#KOR=signal.fftconvolve(SIG1, SIG2[::-1], mode='same')
#FT1 = fft(SIG1) 

#
#AGauss = np.fft.fft(windowGauss, 2048) / (len(windowGauss)/2.0)
#ABartt = np.fft.fft(windowBartt, 2048) / (len(windowBartt)/2.0)
#AHann = np.fft.fft(windowHann, 2048) / (len(windowHann)/2.0)
#ACH = np.fft.fft(windowCH, 2048) / (len(windowCH)/2.0)
#ATuR = np.fft.fft(windowTuR, 2048) / (len(windowTuR)/2.0)
#
#freqGauss = np.linspace(-0.5, 0.5, len(AGauss))
#freqBartt = np.linspace(-0.5, 0.5, len(ABartt))
#freqHann = np.linspace(-0.5, 0.5, len(AHann))
#freqCH = np.linspace(-0.5, 0.5, len(ACH))
#freqTuR = np.linspace(-0.5, 0.5, len(ATuR))
#
#responseGauss  = 20 * np.log10(np.abs(fftshift(AGauss / abs(AGauss).max())))
#responseBartt = 20 * np.log10(np.abs(fftshift(ABartt / abs(ABartt).max())))
#responseHann = np.abs(fftshift(AHann / abs(AHann).max()))
#responseHann = 20 * np.log10(np.maximum(responseHann, 1e-10))
#responseCH = 20 * np.log10(np.abs(fftshift(ACH / abs(ACH).max())))
#responseTuR = 20 * np.log10(np.abs(fftshift(ATuR  / abs(ATuR ).max())))



T=np.linspace(0,len(SIG),len(SIG))/len(SIG)
freq=np.linspace(0,len(FTSIG),len(FTSIG))#-len(FTSIG)/2


F50=plt.figure(50)
F50=plt.subplot(4, 1, 1 ) 
plt.title('FFT izmerjenega signala '+val[V]+'.dat za različne okenske funkcije,  N='+str(LS),fontsize=16)
plt.plot(freq,FTSIG,color=barva[3],ls='-',alpha = 0.95,label='brez okna')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq, FTSIG_G, barva[0], alpha = 0.5, label=r'Gauss ($\sigma$='+str(STD)+')')
plt.plot(freq, FTSIG_B, barva[1], alpha = 0.5, label='Bartlett')
plt.plot(freq, FTSIG_H, barva[2], alpha = 0.5, label='Hann')
plt.plot(freq, FTSIG_CH, barva[4], alpha = 0.5, label='Dolph-Chebyshev (100 dB)')
plt.plot(freq, FTSIG_T, barva[5], alpha = 0.95, label='Tukey')
#plt.plot(freq, FTSIG_HCH, barva[0], alpha = 0.95, label='Hann in Dolph-Chebyshev (100 dB)')
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=1)



F50=plt.subplot(4, 1, 2 ) 
plt.title('FFT izmerjenega signala '+val[V]+'.dat za različne okenske funkcije,  N='+str(LS//2),fontsize=16)
plt.plot(freq[0:LS//2]*2,FTSIG_256,color=barva[3],ls='-',alpha = 0.95,label='brez okna')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//2]*2, FTSIG_G_256, barva[0], alpha = 0.5, label=r'Gauss ($\sigma$='+str(STD)+')')
plt.plot(freq[0:LS//2]*2, FTSIG_B_256, barva[1], alpha = 0.5, label='Bartlett')
plt.plot(freq[0:LS//2]*2, FTSIG_H_256, barva[2], alpha = 0.5, label='Hann')
plt.plot(freq[0:LS//2]*2, FTSIG_CH_256, barva[4], alpha = 0.5, label='Dolph-Chebyshev (100 dB)')
plt.plot(freq[0:LS//2]*2, FTSIG_T_256, barva[5], alpha = 0.95, label='Tukey')
#plt.plot(freq[0:LS//2]*2, FTSIG_HCH_256, barva[0], alpha = 0.95, label='Hann in Dolph-Chebyshev (100 dB)')
#plt.plot(freq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=1)

F50=plt.subplot(4, 1, 3 ) 
plt.title('FFT izmerjenega signala '+val[V]+'.dat za različne okenske funkcije,  N='+str(LS//4),fontsize=16)
plt.plot(freq[0:LS//4]*4,FTSIG_128,color=barva[3],ls='-',alpha = 0.95,label='brez okna')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//4]*4, FTSIG_G_128, barva[0], alpha = 0.5, label=r'Gauss ($\sigma$='+str(STD)+')')
plt.plot(freq[0:LS//4]*4, FTSIG_B_128, barva[1], alpha = 0.5, label='Bartlett')
plt.plot(freq[0:LS//4]*4, FTSIG_H_128, barva[2], alpha = 0.5, label='Hann')
plt.plot(freq[0:LS//4]*4, FTSIG_CH_128, barva[4], alpha = 0.5, label='Dolph-Chebyshev (100 dB)')
plt.plot(freq[0:LS//4]*4, FTSIG_T_128, barva[5], alpha = 0.95, label='Tukey')
#plt.plot(freq[0:LS//4]*4, FTSIG_HCH_128, barva[0], alpha = 0.95, label='Hann in Dolph-Chebyshev (100 dB)')
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,len(SIG)/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=1)


F50=plt.subplot(4, 1, 4 ) 
plt.title('FFT izmerjenega signala '+val[V]+'.dat za različne okenske funkcije,  N='+str(LS//8),fontsize=16)
plt.plot(freq[0:LS//8]*8,FTSIG_64,color=barva[3],ls='-',alpha = 0.95,label='brez okna')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq[0:LS//8]*8, FTSIG_G_64, barva[0], alpha = 0.5, label=r'Gauss ($\sigma$='+str(STD)+')')
plt.plot(freq[0:LS//8]*8, FTSIG_B_64, barva[1], alpha = 0.5, label='Bartlett')
plt.plot(freq[0:LS//8]*8, FTSIG_H_64, barva[2], alpha = 0.5, label='Hann')
plt.plot(freq[0:LS//8]*8, FTSIG_CH_64, barva[4], alpha = 0.5, label='Dolph-Chebyshev (100 dB)')
plt.plot(freq[0:LS//8]*8, FTSIG_T_64, barva[5], alpha = 0.95, label='Tukey')
#plt.plot(freq[0:LS//6]*6, FTSIG_HCH_64, barva[0], alpha = 0.95, label='Hann in Dolph-Chebyshev (100 dB)')
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,len(SIG)/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=1)

################################################################################
#################  okenske funkcije in odzivi #################################
################################################################################
#
#windowGauss = signal.gaussian(51, std=6)
#windowBartt = signal.bartlett(51)
#windowHann = signal.hann(51)
#windowCH = signal.chebwin(51, at=100)
#windowTuR = signal.tukey(51)
#
#AGauss = np.fft.fft(windowGauss, 2048) / (len(windowGauss)/2.0)
#ABartt = np.fft.fft(windowBartt, 2048) / (len(windowBartt)/2.0)
#AHann = np.fft.fft(windowHann, 2048) / (len(windowHann)/2.0)
#ACH = np.fft.fft(windowCH, 2048) / (len(windowCH)/2.0)
#ATuR = np.fft.fft(windowTuR, 2048) / (len(windowTuR)/2.0)
#
#freqGauss = np.linspace(-0.5, 0.5, len(AGauss))
#freqBartt = np.linspace(-0.5, 0.5, len(ABartt))
#freqHann = np.linspace(-0.5, 0.5, len(AHann))
#freqCH = np.linspace(-0.5, 0.5, len(ACH))
#freqTuR = np.linspace(-0.5, 0.5, len(ATuR))
#
#responseGauss  = 20 * np.log10(np.abs(fftshift(AGauss / abs(AGauss).max())))
#responseBartt = 20 * np.log10(np.abs(fftshift(ABartt / abs(ABartt).max())))
#responseHann = np.abs(fftshift(AHann / abs(AHann).max()))
#responseHann = 20 * np.log10(np.maximum(responseHann, 1e-10))
#responseCH = 20 * np.log10(np.abs(fftshift(ACH / abs(ACH).max())))
#responseTuR = 20 * np.log10(np.abs(fftshift(ATuR  / abs(ATuR ).max())))
#
##FT2 = np.ndarray.conjugate(fft(SIG2))
##KOR=ifft(FT1*(FT2))  
#
#F1=plt.figure(1)
#F1=plt.subplot(1, 2, 1 ) 
#plt.plot(windowGauss, barva[0], label=r'Gauss ($\sigma$=6)')
#plt.plot(windowBartt, barva[1], label='Bartlett')
#plt.plot(windowHann, barva[2], label='Hann')
#plt.plot(windowCH, barva[3], label='Dolph-Chebyshev (100 dB)')
#plt.plot(windowTuR, barva[4], label='Turkey')
#plt.title("Okenske funkcije",fontsize=16)
#plt.ylabel("Amplituda",fontsize=16)
#plt.xlabel("Signal",fontsize=16)
#plt.legend(loc=0)
#
#
#F1=plt.subplot(1, 2, 2 ) 
#plt.plot(freqGauss, responseGauss, barva[0], alpha = 0.5, label=r'Gauss ($\sigma$=6)')
#plt.plot(freqBartt, responseBartt, barva[1], alpha = 0.5, label='Bartlett')
#plt.plot(freqHann, responseHann, barva[2], alpha = 0.5, label='Hann')
#plt.plot(freqCH, responseCH, barva[3], alpha = 0.5, label='Dolph-Chebyshev (100 dB)')
#plt.plot(freqTuR, responseTuR, barva[4], alpha = 0.95, label='Turkey')
#plt.axis([-0.5, 0.5, -120, 0])
#plt.title("Frekvenčni odziv na okenske funkcije",fontsize=16)
#plt.ylabel("Normalizirana magnituda I [dB]",fontsize=16)
#plt.xlabel(r'Normalizirana frekvenca $\omega$ [cikel/signal]',fontsize=16)
#plt.legend(loc=0)
#
################################################################################
################################################################################
################################################################################


