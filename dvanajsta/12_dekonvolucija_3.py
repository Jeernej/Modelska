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

barva=['k','r','b','y','m','c']





#### signali in njihove FT

DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/dvanajsta/'
val = ["signal0","signal1","signal2","signal3"]

SIG0 = loadtxt(DIR + val[0]+".dat") # branje
SIG1 = loadtxt(DIR + val[1]+".dat") # branje
SIG2 = loadtxt(DIR + val[2]+".dat") # branje
SIG3 = loadtxt(DIR + val[3]+".dat") # branje
LS= len(SIG0) # dolžina  signala



#FTSIG0 = np.fft.fft(SIG0)
FTSIG0 = abs(np.fft.fft(SIG0))

MIN=0
for i in range(1,len(FTSIG0)//2):
    if FTSIG0[i]<FTSIG0[MIN]:
        MIN=i  ### pozicija minimuma v spektru čistega sigala, kjer naj bi se začel visokofrekvenčni šum

FTSIG0 = np.fft.fft(SIG0)
FTSIG1 = np.fft.fft(SIG1)
FTSIG2 = np.fft.fft(SIG2)
FTSIG3 = np.fft.fft(SIG3)

T=np.linspace(0,len(SIG0),len(SIG0)) ## vektor časovnih točk za plotanje
freq=np.linspace(0,len(FTSIG0),len(FTSIG0)) ## vektor frekvenčnih točk za plotanje


a=1/16
FFTprenosna=2*abs(a)/(abs(a)**2+(T)**2)/(2*a)*32
antiFFTprenosna=np.zeros(len(SIG0))
for i in range(0,len(SIG0)):
    antiFFTprenosna[-i]=FFTprenosna[i]  ### spektri s porezanim visokofrekvenčnim šumom
fftprenosna=antiFFTprenosna+FFTprenosna


FTSIG0_cut =  np.ones(len(FTSIG0))*FTSIG0 ### spektri s porezanim visokofrekvenčnim šumom
FTSIG1_cut =  np.ones(len(FTSIG1))*FTSIG1
FTSIG2_cut =  np.ones(len(FTSIG2))*FTSIG2
FTSIG3_cut =  np.ones(len(FTSIG3))*FTSIG3

for i in range(MIN,len(SIG0)):
#    FTSIG0_cut[i]=FTSIG0[i]  ### spektri s porezanim visokofrekvenčnim šumom
#    FTSIG0_cut[-i]=FTSIG0[-i]  ### spektri s porezanim visokofrekvenčnim šumom
    FTSIG1_cut[i]=FTSIG0[i]
    FTSIG1_cut[-i]=FTSIG0[-i]
    FTSIG2_cut[i]=FTSIG0[i]
    FTSIG2_cut[-i]=FTSIG0[-i]
    FTSIG3_cut[i]=FTSIG0[i]
    FTSIG3_cut[-i]=FTSIG0[-i]


##########################
### Wienerjevi FILTRI iz ocene povprečja šumov (glede na graf začnem pri 100)
##########################

FTsum0=np.average(abs(FTSIG0[100:250])) ##  povprečna vrednost šuma
FTsum1=np.average(abs(FTSIG1[100:250]))
FTsum2=np.average(abs(FTSIG2[100:250]))
FTsum3=np.average(abs(FTSIG3[100:250]))

FTSUM0=np.ones(len(FTSIG0))*FTsum0 ##  povprečni šum /vektor v frekvenčnem/
FTSUM1=np.ones(len(FTSIG1))*FTsum1
FTSUM2=np.ones(len(FTSIG2))*FTsum2
FTSUM3=np.ones(len(FTSIG3))*FTsum3


#FILT1=(abs(np.fft.fft(SIG1))**2-abs(FTSUM1)**2)/abs(np.fft.fft(SIG1))**2 ## Wienerjevi filtri
#FILT2=(abs(np.fft.fft(SIG2))**2-abs(FTSUM2)**2)/abs(np.fft.fft(SIG2))**2
#FILT3=(abs(np.fft.fft(SIG3))**2-abs(FTSUM3)**2)/abs(np.fft.fft(SIG3))**2

FILT1=abs((abs(np.fft.fft(SIG0))**2-abs(FTSUM1)**2)/abs(np.fft.fft(SIG0))**2)**(-1)## Wienerjevi filtri
FILT2=abs((abs(np.fft.fft(SIG0))**2-abs(FTSUM2)**2)/abs(np.fft.fft(SIG0))**2)**(-1)
FILT3=abs((abs(np.fft.fft(SIG0))**2-abs(FTSUM3)**2)/abs(np.fft.fft(SIG0))**2)**(-1)

#
#FILT1=signal.wiener(FTSIG1)
#FILT2=signal.wiener(FTSIG2)
#FILT3=signal.wiener(FTSIG3)
##
#FILT1=(abs(np.fft.fft(SIG0))-abs(FTSUM1))/abs(np.fft.fft(SIG0))
#FILT2=(abs(np.fft.fft(SIG0))-abs(FTSUM2))/abs(np.fft.fft(SIG0))
#FILT3=(abs(np.fft.fft(SIG0))-abs(FTSUM3))/abs(np.fft.fft(SIG0))



########################## SIGNALI #####################################


F50=plt.figure(50)
F50=plt.subplot(2, 1, 1 ) 
plt.title(r'Izmerjeni časovni poteki $c(t)$',fontsize=16)
plt.plot(T,SIG0,color=barva[0],ls='-',alpha = 0.95, label=val[0]+r'.dat = $s(t)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,SIG1,color=barva[1],ls='-',alpha = 0.75, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,SIG2,color=barva[2],ls='-',alpha = 0.55, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,SIG3,color=barva[3],ls='-',alpha = 0.75, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,100*e**(-abs(T)/16)/32,color=barva[4],ls=':',alpha = 0.75, label=r'r(t)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[0:LS//2],SIG_256,color=barva[1],ls=':',alpha = 0.5, label='N='+str(LS//2)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[0:LS//4],SIG_128,color=barva[2],ls=':',alpha = 0.5, label='N='+str(LS//4)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[0:LS//6],SIG_64,color=barva[3],ls=':',alpha = 0.5, label='N='+str(LS//6)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
#plt.yscale('log')
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.xlim([0,len(SIG0)])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=1)


#FTSIG0 = abs(np.fft.fft(SIG0))
#FTSIG1 = abs(np.fft.fft(SIG1))
#FTSIG2 = abs(np.fft.fft(SIG2))
#FTSIG3 = abs(np.fft.fft(SIG3))

F50=plt.subplot(2, 1, 2 ) 
plt.title(r'FFT izmerjenih časovnih potekov $C(\omega)$',fontsize=16)
plt.plot(freq,abs(np.fft.fft(SIG0)),color=barva[0],ls='-',alpha = 0.95, label=val[0]+r'.dat = $S(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(np.fft.fft(SIG1)),color=barva[1],ls='-',alpha = 0.75, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(np.fft.fft(SIG2)),color=barva[2],ls='-',alpha = 0.55, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(np.fft.fft(SIG3)),color=barva[3],ls='-',alpha = 0.75, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSUM0,color=barva[0],ls=':',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSUM1,color=barva[1],ls=':',alpha = 0.75)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSUM2,color=barva[2],ls=':',alpha = 0.55)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FTSUM3,color=barva[3],ls=':',alpha = 0.75)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,2*abs(np.fft.fft(e**(-abs(T)/16)/32))**2 ,ls=':',alpha = 0.75, label=r'R(t)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.xlim([0,LS])
#    plt.ylim([0,250])
plt.legend(loc=1)
#
############################ PRENOSNA ###################################
a=1/16
FFTprenosna=2*abs(a)/(abs(a)**2+(T)**2)/(2*a)*32
#FFTprenosna=np.fft.fft(e**(-T/16)/32)

antiFFTprenosna=np.zeros(len(SIG0))
for i in range(0,len(SIG0)):
    antiFFTprenosna[-i]=FFTprenosna[i]  ### spektri s porezanim visokofrekvenčnim šumom

fftprenosna=antiFFTprenosna+FFTprenosna
#fftprenosna=fftprenosna/sum(fftprenosna)

#fftprenosna=np.zeros(len(SIG0))
#fftprenosna[0:len(SIG0)//2]=FFTprenosna[0:len(SIG0)//2]
#fftprenosna[len(SIG0)//2:len(SIG0)]=antiFFTprenosna[len(SIG0)//2:len(SIG0)]

windowHann = signal.hann(LS)
FTSIG_H = np.fft.fft(SIG0*signal.hann(LS))


#dec_SIG0= np.fft.ifft(np.fft.fft(SIG0)/sum(np.fft.fft(SIG0))/fftprenosna*sum(fftprenosna)) 
dec_SIG0= np.fft.ifft(np.fft.fft(SIG0)/fftprenosna) 
#dec_SIG0= abs(np.fft.ifft(np.fft.fft(SIG0)/fftprenosna) )


T_pr=np.linspace(0,len(SIG0)//2,len(SIG0)//2) ## vektor časovnih točk za plotanje
T_prm=np.linspace(-len(SIG0)//2,0,len(SIG0)//2) ## vektor časovnih točk za plotanje
F60=plt.figure(60)
#F60=plt.subplot(2, 1, 1 ) 
F60=plt.subplot(1, 2, 1 ) 
plt.title(r'Prenosna funkcija merilne naprave $r(t)$',fontsize=16)# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
plt.plot(T_pr,e**(-abs(T_pr)/16)/32,color='k',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T_prm,e**(-abs(T_prm)/16)/32,color='r',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)

#F60=plt.subplot(2, 1, 2 ) 
F60=plt.subplot(1, 2, 2 ) 
plt.title(r'FFT prenosne funkcije merilne naprave $R(\omega)$',fontsize=16)
#plt.plot(freq,np.fft.fft(e**(-abs(T)/16)/32) ,color='k',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FFTprenosna ,color='k',ls=':',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,antiFFTprenosna ,color='r',ls=':',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,fftprenosna ,color='k',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))plt.plot(freq,fftprenosna ,color='r',ls='--',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.yscale('log')
plt.xlim([0,LS])
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)


######################### DEKONVoLUCIJA ###################################
#
#
#
#T_pren=np.linspace(0,26,150) ## čas zaizvrednotenje prenosne funkcije
#u=e**(-abs(T_pren)/16)/32 ## prenosna funkcija
#print(u.min()) 
#recovered, remainder = signal.deconvolve(SIG0,u)
##deconv,  _ = scipy.signal.deconvolve( filtered, prenosna )
##the deconvolution has n = len(signal) - len(gauss) + 1 points
#n = len(SIG0)-len(u)+1
## so we need to expand it by 
#s = (len(SIG0)-n)//2
##on both sides.
#deconv_res = np.zeros(len(SIG0))
#print(len(deconv_res[s:len(SIG0)-s-1]))
#print(len(recovered))
#deconv_res[s:len(SIG0)-s-1] = recovered
## now deconv contains the deconvolution 
## expanded to the original shape (filled with zeros) 




F70=plt.figure(70)
#F70=plt.subplot(3, 1, 1 ) 
F60=plt.subplot(2, 1, 1 ) 
plt.title(r'FFT prenosne funkcije $U(\omega)$ čistega signala in Wienerjevi filtri za zašumljene signale',fontsize=16)
#plt.plot(freq,abs(FTSUM1) ,color='k',ls='-',alpha = 0.95, label=r'$\bar{šum}$ $\tt{signal1.dat}$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(FTSIG0) ,color='k',ls='--',alpha = 0.95, label=r'$C(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(fftprenosna) ,color='k',ls=':',alpha = 0.95, label=r'$R(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(FTSIG0)/abs(fftprenosna) ,color='k',ls='-',alpha = 0.95, label=r'$U(\omega)=C(\omega)/R(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FILT1 ,color=barva[1],ls='-',alpha = 0.95, label=r'Wiener $\phi_1(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FILT2 ,color=barva[2],ls='-',alpha = 0.95, label=r'Wiener $\phi_2(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,FILT3 ,color=barva[3],ls='-',alpha = 0.95, label=r'Wiener $\phi_3(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.yscale('log')
plt.xlim([0,LS//2])
plt.ylabel(r'PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.legend(loc=1)


#F70=plt.subplot(3,1, 2 ) 
F70=plt.subplot(2, 1, 2 ) 
plt.title(r'Dekonvolirane funkcije signalov',fontsize=16)# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
plt.plot(T,SIG0 ,color='k',ls=':',alpha = 0.95,label=r'$s(t)$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,abs( np.fft.ifft(np.fft.fft(SIG0)/np.fft.fft(e**(-abs(T)/16)/32)) ) ,color='k',ls='-',alpha = 0.95,label=r'$u(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,abs( np.fft.ifft(np.fft.fft(SIG0)/2/abs(FFTprenosna)**2 )) ,color='k',ls=':',alpha = 0.95,label=r'$u(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,dec_SIG0,color='k',ls='-',alpha = 0.95,label=r'$u(t)_0$ ;  '+val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,abs( np.fft.ifft(FTSIG0/2/abs(np.fft.fft(e**(-abs(T)/16)/32))**2) ) ,color='y',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,deconv_res,color='b',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,signal.deconvolve(SIG0,u)[0],color='b',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,signal.deconvolve(FTSIG0,np.fft.fft(u))[1],color='g',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.ylabel('Amplituda' ,fontsize=16)  
#plt.xlabel(' t ' ,fontsize=16)
#plt.xlim([0,LS])
##plt.yscale('log')
#plt.legend(loc=1)


#F70=plt.subplot(3,1, 3 ) 
#plt.title(r'Dekonvolirana prenosna funkcija zašumljenih $\tt{signal1,2,3.dat}$ signala $\widetilde{u}(t)$')# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
##plt.plot(T,SIG0 ,color='k',ls='-',alpha = 0.95, label=val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T,abs( np.fft.ifft(FILT1*np.fft.fft(SIG1)/np.fft.fft(e**(-abs(T)/16)/32)) ) ,color=barva[1],ls='-',alpha = 0.95, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T,abs( np.fft.ifft(FILT2*np.fft.fft(SIG2)/np.fft.fft(e**(-abs(T)/16)/32)) ) ,color=barva[2],ls='-',alpha = 0.95, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T,abs( np.fft.ifft(FILT3*np.fft.fft(SIG3)/np.fft.fft(e**(-abs(T)/16)/32)) ) ,color=barva[3],ls='-',alpha = 0.95, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT1*np.fft.fft(SIG1)/fftprenosna)  ,color=barva[1],ls='-',alpha = 0.95, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT2*np.fft.fft(SIG2)/fftprenosna)  ,color=barva[2],ls='-',alpha = 0.95, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT3*np.fft.fft(SIG3)/fftprenosna)  ,color=barva[3],ls='-',alpha = 0.95, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.ylabel('Amplituda' ,fontsize=16)  
#plt.xlabel(' t ' ,fontsize=16)
#plt.xlim([0,LS])
##plt.yscale('log')
#plt.legend(loc=1)

#F70=plt.subplot(3,1, 3 ) 
#F1=plt.subplot(1,3, 3 ) 
#plt.title(r'Dekonvolirana prenosna funkcija zašumljenih signalov $\widetilde{u}(t)$')# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
#plt.plot(T,SIG0 ,color='k',ls='-',alpha = 0.95, label=val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT1*FTSIG1/fftprenosna)  ,color=barva[1],ls='-',alpha = 0.95, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT2*FTSIG2/fftprenosna)  ,color=barva[2],ls='-',alpha = 0.95, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT3*FTSIG3/fftprenosna)  ,color=barva[3],ls='-',alpha = 0.95, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FILT1*FTSIG1_cut/fftprenosna)  ,color=barva[1],ls='-',alpha = 0.95, label='$\widetilde{u}_1(t)$ ;  '+val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FILT2*FTSIG2_cut/fftprenosna)  ,color=barva[2],ls='-',alpha = 0.95, label='$\widetilde{u}_2(t)$ ;  '+val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FILT3*FTSIG3_cut/fftprenosna)  ,color=barva[3],ls='-',alpha = 0.95, label='$\widetilde{u}_3(t)$ ;  '+val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.xlim([0,LS])
#plt.xlim([0,LS/2])
#plt.yscale('log')
plt.legend(loc=1)

##################################################################



F1=plt.figure(1)
F1=plt.subplot(3, 1, 1 ) 
#F1=plt.subplot(1, 3, 1 ) 
plt.title(r'FFT prenosne funkcije signala $U(\omega)$',fontsize=16)
#plt.plot(freq,FTSUM1 ,color='b',ls='-',alpha = 0.95, label=r'$\bar{šum}$ $\tt{signal1.dat}$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FILT1 ,color='g',ls='-',alpha = 0.95, label=r'Wiener $\phi_1$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(FTSIG0_cut) ,color='k',ls='--',alpha = 0.95, label=r'$U(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(FTSIG1_cut),color='k',ls='--',alpha = 0.95, label=r'$U(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(freq,abs(FTSIG2_cut) ,color='k',ls='--',alpha = 0.95, label=r'$U(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,2*abs(np.fft.fft(e**(-abs(T)/16)/32))**2 ,color='k',ls=':',alpha = 0.95, label=r'$R(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG0_cut/2*abs(np.fft.fft(e**(-abs(T)/16)/32))**2 ,color='k',ls='-',alpha = 0.95, label=r'$U(\omega)/R(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xlim([0,LS])
plt.yscale('log')
#plt.xlim([0,LS/2])
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
plt.legend(loc=1)

#recovered, remainder = signal.deconvolve(SIG0,e**(-abs(SIG0)/16)/32)
#u=e**(-abs(T)/16)/32

F1=plt.subplot(3,1, 2 ) 
#F1=plt.subplot(1, 3, 2 ) 
plt.title(r'Dekonvolirana prenosna funkcija signala $u(t)$')# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
plt.plot(T,SIG0 ,color='r',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FTSIG0_cut/fftprenosna) ,color='k',ls='-',alpha = 0.95,label=r'$u(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,abs( np.fft.ifft(FTSIG0/2/abs(np.fft.fft(e**(-abs(T)/16)/32))**2) ) ,color='y',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,signal.deconvolve(SIG0,u)[1],color='b',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,signal.deconvolve(FTSIG0,np.fft.fft(u))[1],color='g',ls='-',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.xlim([0,LS])
#plt.xlim([0,LS/2])
#plt.yscale('log')
plt.legend(loc=1)

F1=plt.subplot(3,1, 3 ) 
#F1=plt.subplot(1,3, 3 ) 
plt.title(r'Dekonvolirana prenosna funkcija $\tt{signal1.dat}$ signala $\widetilde{u}(t)$')# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
#plt.plot(T,SIG0 ,color='k',ls='-',alpha = 0.95, label=val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT1*FTSIG1/fftprenosna)  ,color=barva[1],ls='-',alpha = 0.95, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT2*FTSIG2/fftprenosna)  ,color=barva[2],ls='-',alpha = 0.95, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T, np.fft.ifft(FILT3*FTSIG3/fftprenosna)  ,color=barva[3],ls='-',alpha = 0.95, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FILT1*FTSIG1_cut/fftprenosna)  ,color=barva[1],ls='-',alpha = 0.95, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FILT2*FTSIG2_cut/fftprenosna)  ,color=barva[2],ls='-',alpha = 0.95, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T, np.fft.ifft(FILT3*FTSIG3_cut/fftprenosna)  ,color=barva[3],ls='-',alpha = 0.95, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
plt.ylabel('Amplituda' ,fontsize=16)  
plt.xlabel(' t ' ,fontsize=16)
plt.xlim([0,LS])
#plt.xlim([0,LS/2])
#plt.yscale('log')
plt.legend(loc=1)
