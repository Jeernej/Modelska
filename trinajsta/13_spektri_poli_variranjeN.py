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


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/trinajsta/'
val = ["val2.dat","val3.dat","co2.dat","luna.dat","borza.dat","Wolf_number.dat"]

SIG = loadtxt(DIR + val[0]) # branje
SIG2 = loadtxt(DIR + val[1]) # branje
co=loadtxt(DIR + val[2]) # branje
SIG3=[float(co[i][1]) for i in range(len(co)) if float(co[i][1])!=-99.99] 


LS= len(SIG) # dolžina  signala
LS3= len(SIG3) # dolžina  signala

T=np.linspace(0,len(SIG),len(SIG)) /len(SIG)
T3=[float(co[i][0]) for i in range(len(co)) if float(co[i][1])!=-99.99]

SIG3_lin=np.zeros(len(SIG3))
T3=np.array([i for i in range(len(SIG3))])
k=np.polyfit(T3,SIG3,1)
for i in range(len(SIG3)):
    SIG3_lin[i]=SIG3[i]-(k[0]*i+k[1])


SIG_256= SIG[0:LS//2]
SIG_128= SIG[0:LS//4]

SIG2_256= SIG2[0:LS//2]
SIG2_128= SIG2[0:LS//4]

SIG3_lin_302= SIG3_lin[0:LS3//2]
SIG3_lin_151= SIG3_lin[0:LS3//4]

########### FOURIERE ############################

FTSIG = 2*abs(np.fft.fft(SIG))**2 
FTSIG_CH = 2*abs(np.fft.fft(SIG*signal.chebwin(LS, at=100)))**2 
FTSIG_256 = 2*abs(np.fft.fft(SIG_256))**2 
FTSIG_CH_256 = 2*abs(np.fft.fft(SIG_256*signal.chebwin(LS//2, at=100)))**2 
FTSIG_128 = 2*abs(np.fft.fft(SIG_128))**2  
FTSIG_CH_128 = 2*abs(np.fft.fft(SIG_128*signal.chebwin(LS//4, at=100)))**2 

FTSIG2 = 2*abs(np.fft.fft(SIG2))**2 
FTSIG2_CH = 2*abs(np.fft.fft(SIG2*signal.chebwin(LS, at=100)))**2 
FTSIG2_256 = 2*abs(np.fft.fft(SIG2_256))**2 
FTSIG2_CH_256 = 2*abs(np.fft.fft(SIG2_256*signal.chebwin(LS//2, at=100)))**2 
FTSIG2_128 = 2*abs(np.fft.fft(SIG2_128))**2  
FTSIG2_CH_128 = 2*abs(np.fft.fft(SIG2_128*signal.chebwin(LS//4, at=100)))**2 

FTSIG3_lin = 2*abs(np.fft.fft(SIG3_lin))**2 
FTSIG3_lin_CH = 2*abs(np.fft.fft(SIG3_lin*signal.chebwin(LS3, at=100)))**2 
FTSIG3_lin_302 = 2*abs(np.fft.fft(SIG3_lin_302))**2 
FTSIG3_lin_CH_302 = 2*abs(np.fft.fft(SIG3_lin_302*signal.chebwin(LS3//2, at=100)))**2 
FTSIG3_lin_151 = 2*abs(np.fft.fft(SIG3_lin_151))**2  
FTSIG3_lin_CH_151 = 2*abs(np.fft.fft(SIG3_lin_151*signal.chebwin(LS3//4, at=100)))**2 

freq=np.linspace(0,len(FTSIG),len(FTSIG))#-len(FTSIG)/2
freq3=np.linspace(0,len(FTSIG3_lin),len(FTSIG3_lin))#-len(FTSIG)/2


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
#    return linalg.solve_toeplitz(c, b)  ### solving the Yule-Walker Equations with 

def z(ak):
    Npoli=len(ak)
    p=[ak[Npoli-1-i] if i<Npoli else 1 for i in range(Npoli+1)]
    return (np.roots(p))**-1  # kompleksne Z rešitve za   p[0] * Z**n + p[1] * Z**(n-1) + ... + p[n-1]*Z + p[n]=

def z_in(zk):
    z_in=np.ones(len(zk))*zk
    for i in range(len(zk)) :
        if np.sqrt(zk.real[i]**2+zk.imag[i]**2)>1:
            z_in.real[i]=zk.real[i]-2*(zk.real[i]-(zk.real[i]/abs(zk[i])))   ### preslikamo Z-je v enotski krog
            z_in.imag[i]=zk.imag[i]-2*(zk.imag[i]-(zk.imag[i]/abs(zk[i]))) ### preslikamo Z-je v enotski krog           
    return z_in  # kompleksne Z rešitve za   p[0] * Z**n + p[1] * Z**(n-1) + ... + p[n-1]*Z + p[n]=


def ak_abs(zk,ak):
    Npoli=len(ak)
    z=z_in(zk)
    for i in range(len(z)) : z[i]=1/z[i]
    rešitev=np.poly1d(z, True)                           ### iskanje novih ak-jev z normiranimi Z-ji
    return [rešitev[i+1]/rešitev[0] for i in range(Npoli)]

def P(ni,ak,N): ##### računanje spektra PSD
    Npoli=len(ak)
    return 1/(abs(1+sum(ak[i]*np.exp(-1j*ni*2*np.pi*(i+1)/(N/2)) for i in range(Npoli)))**2)  

##################################################################


def S(ak, signal):
    Stil=[signal[i] for i in range(int(len(signal)/2))]
    N=len(Stil)
    p=len(ak)
    for n in range(int(len(signal)/2)-p):
        Stil.append(sum(-ak[i]*Stil[n+N-i-1] for i in range(p)))
    return Stil


##################################################################

## val2
#
#F50=plt.figure(50)
#F50=plt.subplot(3, 1, 1 ) 
#plt.title('Spekter signala '+val[0]+' za različne metode,  N='+str(LS),fontsize=16)
#plt.plot(freq,FTSIG,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq, FTSIG_CH, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')
#
#ali2=a(SIG,8)
#zk2=z(ali2)
#alin2=ak_abs(zk2,ali2)
##plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3))],color=barva[0],ls='--',label='p={}'.format(8))
#plt.plot(freq*2,[P(i,alin2,int(LS)) for i in range(int(LS))],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(8))
#
#plt.yscale('log')
#plt.ylabel('PSD($\omega$)' ,fontsize=16)   
#plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,LS/2])
#plt.legend(loc=1)
#
#
#
#F50=plt.subplot(3, 1, 2 ) 
#plt.title('Spekter signala '+val[0]+' za različne metode,  N='+str(LS//2),fontsize=16)
#plt.plot(freq[0:LS//2]*2,FTSIG_256,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//2]*2, FTSIG_CH_256, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')
#
#ali2=a(SIG_256,8)
#zk2=z(ali2)
#alin2=ak_abs(zk2,ali2)
##plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],ls='--',label='p={}'.format(8))
#plt.plot(freq[0:LS//2]*2,[P(i,alin2,int(LS)) for i in range(int(LS)//2)],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(8))
#
#plt.yscale('log')
#plt.ylabel('PSD($\omega$)' ,fontsize=16)   
#plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,LS/2])
#plt.legend(loc=1)
#
#F50=plt.subplot(3, 1, 3 ) 
#plt.title('Spekter signala '+val[0]+' za različne metode,  N='+str(LS//4),fontsize=16)
#plt.plot(freq[0:LS//4]*4,FTSIG_128,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//4]*4, FTSIG_CH_128, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')
#
#ali2=a(SIG_128,8)
#zk2=z(ali2)
#alin2=ak_abs(zk2,ali2)
##plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],ls='--',label='p={}'.format(8))
#plt.plot(freq[0:LS//4]*2,[P(i,alin2,int(LS)) for i in range(int(LS)//4)],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(8))
#
#plt.yscale('log')
#plt.ylabel('PSD($\omega$)' ,fontsize=16)   
#plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,LS/2])
#plt.legend(loc=1)


#################################################################


## val3

#
#
#F50=plt.figure(50)
#F50=plt.subplot(3, 1, 1 ) 
#plt.title('Spekter signala '+val[1]+' za različne metode,  N='+str(LS),fontsize=16)
#plt.plot(freq,FTSIG2,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq, FTSIG2_CH, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')
#
#ali2=a(SIG2,10)
#zk2=z(ali2)
#alin2=ak_abs(zk2,ali2)
##plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3))],color=barva[0],ls='--',label='p={}'.format(8))
#plt.plot(freq*2,[P(i,alin2,int(LS)) for i in range(int(LS))],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(8))
#
#plt.yscale('log')
#plt.ylabel('PSD($\omega$)' ,fontsize=16)   
#plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,LS/2])
#plt.legend(loc=4)
#
#
#
#F50=plt.subplot(3, 1, 2 ) 
#plt.title('Spekter signala '+val[1]+' za različne metode,  N='+str(LS//2),fontsize=16)
#plt.plot(freq[0:LS//2]*2,FTSIG2_256,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//2]*2, FTSIG2_CH_256, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')
#
#ali2=a(SIG2_256,10)
#zk2=z(ali2)
#alin2=ak_abs(zk2,ali2)
##plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],ls='--',label='p={}'.format(8))
#plt.plot(freq[0:LS//2]*2,[P(i,alin2,int(LS)) for i in range(int(LS)//2)],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(8))
#
#plt.yscale('log')
#plt.ylabel('PSD($\omega$)' ,fontsize=16)   
#plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,LS/2])
#plt.legend(loc=4)
#
#
#
#F50=plt.subplot(3, 1, 3 ) 
#plt.title('Spekter signala '+val[1]+' za različne metode,  N='+str(LS//4),fontsize=16)
#plt.plot(freq[0:LS//4]*4,FTSIG2_128,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//4]*4, FTSIG2_CH_128, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')
#
#ali2=a(SIG2_128,10)
#zk2=z(ali2)
#alin2=ak_abs(zk2,ali2)
##plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],ls='--',label='p={}'.format(8))
#plt.plot(freq[0:LS//4]*2,[P(i,alin2,int(LS)) for i in range(int(LS)//4)],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(8))
#
#plt.yscale('log')
#plt.ylabel('PSD($\omega$)' ,fontsize=16)   
#plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,LS/2])
#plt.legend(loc=4)




#################################################################


#co2


F50=plt.figure(50)
F50=plt.subplot(3, 1, 1 ) 
plt.title('Spekter signala '+val[2]+' za različne metode,  N='+str(LS3),fontsize=16)
plt.plot(freq3,FTSIG3_lin,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3, FTSIG3_lin_CH, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')

ali2=a(SIG3_lin,14)
zk2=z(ali2)
alin2=ak_abs(zk2,ali2)
#plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3))],color=barva[0],ls='--',label='p={}'.format(8))
plt.plot(freq3*2,[P(i,alin2,int(LS3)) for i in range(int(LS3))],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(10))

plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
plt.ylim([0.01,1000000])
plt.legend(loc=1)



F50=plt.subplot(3, 1, 2 ) 
plt.title('Spekter signala '+val[2]+' za različne metode,  N='+str(LS3//2),fontsize=16)
plt.plot(freq3[0:LS3//2]*2,FTSIG3_lin_302,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3[0:LS3//2]*2, FTSIG3_lin_CH_302, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')

ali2=a(SIG3_lin_302,14)
zk2=z(ali2)
alin2=ak_abs(zk2,ali2)
#plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],ls='--',label='p={}'.format(8))
plt.plot(freq3[0:LS3//2]*2,[P(i,alin2,int(LS3)) for i in range(int(LS3)//2)],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(10))

plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
plt.legend(loc=1)



F50=plt.subplot(3, 1, 3 ) 
plt.title('Spekter signala '+val[2]+' za različne metode,  N='+str(LS3//4),fontsize=16)
plt.plot(freq3[0:LS3//4]*4,FTSIG3_lin_151,color='k',ls='-',alpha = 0.95,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3[0:LS3//4]*4, FTSIG3_lin_CH_151, barva[4], alpha = 0.5, label='FFT + Dolph-Chebyshev (100 dB)')

ali2=a(SIG3_lin_151,14)
zk2=z(ali2)
alin2=ak_abs(zk2,ali2)
#plt.plot(freq*2,[P(i,ali2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],ls='--',label='p={}'.format(8))
plt.plot(freq3[0:LS3//4]*2,[P(i,alin2,int(LS3)) for i in range(int(LS3)//4)],color=barva[0],label='MEM, p={} ; (|z|<1)'.format(10))

plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
plt.legend(loc=1)
