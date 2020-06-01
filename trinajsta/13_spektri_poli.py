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


########### FOURIERE ############################

FTSIG = 2*abs(np.fft.fft(SIG))**2 

FTSIG2 = 2*abs(np.fft.fft(SIG2))**2  

FTSIG3 = 2*abs(np.fft.fft(SIG3))**2 
FTSIG3L = 2*abs(np.fft.fft(SIG3_lin))**2 

FTSIG_H = 2*abs(np.fft.fft(SIG*signal.hann(LS)))**2 
FTSIG_CH = 2*abs(np.fft.fft(SIG*signal.chebwin(LS, at=100)))**2 
FTSIG2_H = 2*abs(np.fft.fft(SIG2*signal.hann(LS)))**2 
FTSIG2_CH = 2*abs(np.fft.fft(SIG2*signal.chebwin(LS, at=100)))**2 
FTSIG3_H = 2*abs(np.fft.fft(SIG3*signal.hann(LS3)))**2 
FTSIG3_CH = 2*abs(np.fft.fft(SIG3*signal.chebwin(LS3, at=100)))**2 
FTSIG3L_H = 2*abs(np.fft.fft(SIG3_lin*signal.hann(LS3)))**2 
FTSIG3L_CH = 2*abs(np.fft.fft(SIG3_lin*signal.chebwin(LS3, at=100)))**2 

freq=np.linspace(0,len(FTSIG),len(FTSIG))#-len(FTSIG)/2
freq3=np.linspace(0,len(FTSIG3),len(FTSIG3))#-len(FTSIG)/2

#
#

F110=plt.figure(110)
F110=plt.subplot(2, 1, 1 ) 
plt.title('Izmerjena signala '+val[0]+' in '+val[1],fontsize=16)
plt.plot(T,SIG,color='c',ls='-',alpha = 0.5, label=val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,SIG2,color='m',ls='-',alpha = 0.5, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.legend(loc=1)

F110=plt.subplot(2, 1, 2 ) 
plt.title('Izmerjen vzorec '+val[2],fontsize=16)
plt.plot(T3,SIG3,color='c',ls='-',alpha = 0.85, label=val[2])#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,k[0]*T3+k[1],color='k',ls='-',alpha = 0.85, label='trend fit')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T3,SIG3_lin,color='m',ls='-',alpha = 0.5, label=val[2]+'; (odštet trend)')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.legend(loc=1)




F10=plt.figure(10)
F10=plt.subplot(3, 1, 1 ) 
plt.title('Izmerjena signala '+val[0]+' in '+val[1],fontsize=16)
plt.plot(T,SIG,color='c',ls='-',alpha = 0.5, label=val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,SIG2,color='m',ls='-',alpha = 0.5, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.legend(loc=1)

F10=plt.subplot(3, 1, 2 ) 
plt.title('FFT izmerjenega signala '+val[0]+' za različna okna ;  N='+str(LS),fontsize=16)
plt.plot(freq,FTSIG,color=barva[1],ls='-',alpha = 0.5,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG_H,color=barva[2],ls='-',alpha = 0.5,label='FFT + Hann  ')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG_CH,color=barva[3],ls='-',alpha = 0.5,label='FFT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG2,color=barva[3],ls='-',alpha = 0.5,label='N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//2]*2,FTSIG_256,color=barva[1],ls='-',alpha = 0.5,label='N='+str(LS//2))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//4]*4,FTSIG_128,color=barva[2],ls='-',alpha = 0.5,label='N='+str(LS//4))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//6]*6,FTSIG_64,color=barva[0],ls='-',alpha = 0.5,label='N='+str(LS//6))#+'{:.{}f}'.format(SHfi, 3 ))
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

F10=plt.subplot(3, 1, 3 ) 
plt.title('FFT izmerjenega signala '+val[1]+' za različna okna ;  N='+str(LS),fontsize=16)
#plt.plot(FTfreq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG2,color=barva[1],ls='-',alpha = 0.5,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG2_H,color=barva[2],ls='-',alpha = 0.5,label='FFT + Hann  ')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSIG2_CH,color=barva[3],ls='-',alpha = 0.5,label='FFT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//2]*2,FTSIG2_256,color=barva[1],ls='-',alpha = 0.5,label='N='+str(LS//2))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//4]*4,FTSIG2_128,color=barva[2],ls='-',alpha = 0.5,label='N='+str(LS//4))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//6]*6,FTSIG2_64,color=barva[0],ls='-',alpha = 0.5,label='N='+str(LS//6))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,len(SIG)/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=1)



F20=plt.figure(20)
F20=plt.subplot(3, 1, 1 ) 
plt.title('Izmerjen vzorec '+val[2],fontsize=16)
plt.plot(T3,SIG3,color='c',ls='-',alpha = 0.85, label=val[3])#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T3,k[0]*T3+k[1],color='k',ls='-',alpha = 0.85, label='trend fit')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T3,SIG3_lin,color='m',ls='-',alpha = 0.5, label=val[3]+'; (odštet trend)')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.legend(loc=1)

F20=plt.subplot(3, 1, 2 ) 
plt.title('FFT izmerjenega signala '+val[2]+' za različna okna ;  N='+str(LS3),fontsize=16)
plt.plot(freq3,FTSIG3,color=barva[1],ls='-',alpha = 0.5,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3_H,color=barva[2],ls='-',alpha = 0.5,label='FFT + Hann  ')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3_CH,color=barva[3],ls='-',alpha = 0.5,label='FFT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG2,color=barva[3],ls='-',alpha = 0.5,label='N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//2]*2,FTSIG_256,color=barva[1],ls='-',alpha = 0.5,label='N='+str(LS//2))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//4]*4,FTSIG_128,color=barva[2],ls='-',alpha = 0.5,label='N='+str(LS//4))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//6]*6,FTSIG_64,color=barva[0],ls='-',alpha = 0.5,label='N='+str(LS//6))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

F20=plt.subplot(3, 1, 3) 
plt.title('FFT izmerjenega signala '+val[2]+' za različna okna in odšteti linearni trend ;  N='+str(LS3),fontsize=16)
plt.plot(freq3,FTSIG3L,color=barva[1],ls='-',alpha = 0.5,label='FFT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3L_H,color=barva[2],ls='-',alpha = 0.5,label='FFT + Hann')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3L_CH,color=barva[3],ls='-',alpha = 0.5,label='FFT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG2,color=barva[3],ls='-',alpha = 0.5,label='N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//2]*2,FTSIG_256,color=barva[1],ls='-',alpha = 0.5,label='N='+str(LS//2))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//4]*4,FTSIG_128,color=barva[2],ls='-',alpha = 0.5,label='N='+str(LS//4))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq[0:LS//6]*6,FTSIG_64,color=barva[0],ls='-',alpha = 0.5,label='N='+str(LS//6))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
plt.ylim([0.001,10000000])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

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



ali2=a(SIG,5)
zk2=z(ali2)
alin2=ak_abs(zk2,ali2)

ali3=a(SIG2,5)
zk3=z(ali3)
alin3=ak_abs(zk3,ali3)


aliCO2=a(SIG3,3)
zkCO2=z(aliCO2)
alinCO2=ak_abs(zkCO2,aliCO2)

##################################################################################

# val2

F50=plt.figure(50)
F50=plt.subplot(1,2, 1 ) 
plt.title('Poli za vzorec  '+val[0]+' ;  N='+str(LS),fontsize=16)
c=0
for i in [2,4,6,8,48]: # števila polov
    ali2=a(SIG,i)
    zk2=z(ali2)
    zk2_in=z_in(zk2)
    plt.scatter(zk2.real,zk2.imag, color=barva[c],alpha = 0.95,  marker='x', label='p={}'.format(i))
#    plt.scatter(zk2_in.real,zk2_in.imag, color=barva[c],alpha = 0.5,   marker='x',label='p={}'.format(i))
    c=c+1
    
circle1 = plt.Circle((0, 0), 1, color=barva[0],fill=False)
F50.add_artist(circle1)

plt.xlabel('Re[z]',fontsize=16)
plt.ylabel('Im[z]',fontsize=16)
plt.legend(loc='best')

F50=plt.subplot(1,2, 2 ) 
plt.title('Spekter vzorca  '+val[0]+' ;  N='+str(LS),fontsize=16)
c=0
for i in [2,4,6,8,48]: # števila polov
    ali2=a(SIG,i)
    zk2=z(ali2)
    alin2=ak_abs(zk2,ali2)
    plt.plot(freq*2,[P(i,ali2,int(LS)) for i in range(int(512))],color=barva[c],ls='--',label='p={}'.format(i))
#    plt.plot(freq*2,[P(i,alin2) for i in range(int(512))],color=barva[c],label='p={} normirano'.format((i+1)*2))
    c=c+1
plt.plot(freq,FTSIG,color='k',ls='-',alpha = 0.95,label='FTT,  N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS/2])
plt.legend(loc='best')
plt.yscale('log')


# val3

F60=plt.figure(60)
F60=plt.subplot(1,2, 1 ) 
plt.title('Poli za vzorec  '+val[1]+' ;  N='+str(LS),fontsize=16)
c=0
for i in [4,6,8,10,48]: # števila polov
    ali3=a(SIG2,i)
    zk3=z(ali3)
    zk3_in=z_in(zk3)
    plt.scatter(zk3.real,zk3.imag, color=barva[c],alpha = 0.95, marker='x', label='p={}'.format(i))
#    plt.scatter(zk3_in.real,zk3_in.imag, color=barva[c],alpha = 0.5,  marker='p',label='p={} ; (|z|<1)'.format(i))
    c=c+1

circle1 = plt.Circle((0, 0), 1, color=barva[0],fill=False)
F60.add_artist(circle1)

plt.xlabel('Re[z]',fontsize=16)
plt.ylabel('Im[z]',fontsize=16)
plt.legend(loc='best')

F60=plt.subplot(1,2, 2 ) 
plt.title('Spekter vzorca  '+val[1]+' ;  N='+str(LS),fontsize=16)
c=0
for i in [4,6,8,10,48]: # števila polov
    ali3=a(SIG2,i)
    zk3=z(ali3)
    alin3=ak_abs(zk3,ali3)
    plt.plot(freq*2,[100*P(i,ali3,int(LS)) for i in range(int(512))],color=barva[c], ls='--',label='p={}'.format(i))
#    plt.plot(freq*2,[100*P(i,alin3) for i in range(int(512))],color=barva[c], label='p={} ; (|z|<1)'.format(i))
    c=c+1

#plt.plot([i for i in range(int(512/2))],[10000*P(i,ali3) for i in range(int(512/2))],label='P($\omega$)')
plt.plot(freq,FTSIG2,color='k',ls='-',alpha = 0.95,label='FTT,  N='+str(LS))#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS/2])
plt.legend(loc='best')
plt.yscale('log')



F90=plt.figure(90)
F90=plt.subplot(1,2, 1 ) 
plt.title('Poli za vzorec  '+val[2]+' ;  N='+str(LS3),fontsize=16)
c=0
for i in [8,10,12,14,48]: # števila polov
    ali4=a(SIG3,i)
    zk4=z(ali4)
    zk4_in=z_in(zk4)
    plt.scatter(zk4.real,zk4.imag, color=barva[c],alpha = 0.95, marker='x', label='p={}'.format(i))
#    plt.scatter(zk4_in.real,zk4_in.imag, color=barva[c],alpha = 0.5,  marker='p',label='p={} ; (|z|<1)'.format(i))
    c=c+1

circle1 = plt.Circle((0, 0), 1, color=barva[0],fill=False)
F90.add_artist(circle1)

plt.xlabel('Re[z]',fontsize=16)
plt.ylabel('Im[z]',fontsize=16)
plt.legend(loc='best')

F90=plt.subplot(1,2, 2 ) 
plt.title('Spekter vzorca  '+val[2]+' ;  N='+str(LS3),fontsize=16)
c=0
for i in [8,10,12,14,48]: # števila polov
    ali4=a(SIG3,i)
    zk4=z(ali4)
    alin4=ak_abs(zk4,ali4)
    plt.plot(freq3*2,[100*P(i,ali4,int(LS3)) for i in range(int(LS3))],color=barva[c], ls='-',label='p={}'.format(i))
#    plt.plot(freq3*2.34,[100*P(i,alin4) for i in range(int(LS3))],color=barva[c], label='p={} ; (|z|<1)'.format(i))
    c=c+1

plt.plot(FTSIG3L,color='k',ls='-',alpha = 0.95,label='FTT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3_H,color='k',ls=':',alpha = 0.5,label='FTT + Hann')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3_CH,color='k',ls='--',alpha = 0.5,label='FTT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
plt.ylim([0.01,10000000])
plt.legend(loc='best')
plt.yscale('log')




F190=plt.figure(190)
F190=plt.subplot(1,2, 1 ) 
plt.title('Poli za vzorec  '+val[2]+' (odštet trend) ;  '+'N='+str(LS3),fontsize=16)
c=0
for i in [8,10,12,14,16]: # števila polov
    ali5=a(SIG3_lin,i)
    zk5=z(ali5)
    zk5_in=z_in(zk5)
    plt.scatter(zk5.real,zk5.imag, color=barva[c],alpha = 0.95, marker='x', label='p={}'.format(i))
#    plt.scatter(zk5_in.real,zk5_in.imag, color=barva[c],alpha = 0.5,  marker='p',label='p={} ; (|z|<1)'.format(i))
    c=c+1

circle1 = plt.Circle((0, 0), 1, color=barva[0],fill=False)
F190.add_artist(circle1)
plt.xlim([-1.5,1.5])
plt.xlabel('Re[z]',fontsize=16)
plt.ylabel('Im[z]',fontsize=16)
plt.legend(loc='best')

F190=plt.subplot(1,2, 2 ) 
plt.title('Spekter vzorca  '+val[2]+' (odštet trend) ;  '+'N='+str(LS3),fontsize=16)
c=0
for i in [8,10,12,14,16]: # števila polov
    ali5=a(SIG3_lin,i)
    zk5=z(ali5)
    alin5=ak_abs(zk5,ali5)
    plt.plot(freq3*2,[100*P(i,ali5,int(LS3)) for i in range(int(LS3))],color=barva[c], ls='-',label='p={}'.format(i))
#    plt.plot(freq3*2.34,[100*P(i,alin5) for i in range(int(LS3))],color=barva[c], label='p={} ; (|z|<1)'.format(i))
    c=c+1


plt.plot(FTSIG3L,color='k',ls='-',alpha = 0.95,label='FTT')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3L_H,color='k',ls=':',alpha = 0.5,label='FTT + Hann')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq3,FTSIG3L_CH,color='k',ls='--',alpha = 0.5,label='FTT + Dolph-Chebyshev (100 dB)')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS3/2])
plt.ylim([0.001,10000000])
plt.legend(loc='best')
plt.yscale('log')

