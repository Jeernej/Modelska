# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 00:51:01 2019

@author: jernej
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:55:47 2019

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

barva=['r','b','y','g','m','c','o']


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/trinajsta/'
val = ["val2.dat","val3.dat","co2.dat","luna.dat","borza.dat","Wolf_number.dat"]

############################
SIG = loadtxt(DIR + val[0]) # branje
############################
SIG2 = loadtxt(DIR + val[1]) # branje
############################
co=loadtxt(DIR + val[2]) # branje
SIG3=[float(co[i][1]) for i in range(len(co)) if float(co[i][1])!=-99.99] 
############################
with open('luna.dat') as f:
    podaci=[l.strip().split(" ") for l in f]
luna=[[float(podaci[i][1]),float(podaci[i][2])] for i in range(len(podaci))]
SIG4=[luna[i][1] for i in range(len(luna))]
############################
SIG5=loadtxt(DIR + val[4]) # BORZA branje OK
############################
WOLF=loadtxt(DIR + val[5]) # WOLF branje
SIG6=[float(WOLF[i][2]) for i in range(len(WOLF))] 
############################

noise=np.random.random(len(SIG2))
SIG_sum1=SIG+noise*5

LS= len(SIG) # dolžina  signala
LS3= len(SIG3) # dolžina  signala
LS4= len(SIG4) # dolžina  signala
LS5= len(SIG5) # dolžina  signala
LS6= len(SIG6) # dolžina  signala

T=np.linspace(0,len(SIG),len(SIG)) /len(SIG)
#T3=[float(co[i][0]) for i in range(len(co)) if float(co[i][1])!=-99.99]#/T3[-1]

SIG3_lin=np.zeros(len(SIG3))
T3=np.array([i for i in range(len(SIG3))])#/len(SIG3)
T4=np.array([i for i in range(len(SIG4))])#/len(SIG3)
T5=np.array([i for i in range(len(SIG5))])#/len(SIG3)
T6=np.array([i for i in range(len(SIG6))])#/len(SIG3)

k=np.polyfit(T3,SIG3,1)
for i in range(len(SIG3)):
    SIG3_lin[i]=SIG3[i]-(k[0]*i+k[1])

SIG_256= SIG[0:LS//2]
SIG_noise1_256= SIG_sum1[0:LS//2]
#SIG_noise2_256= SIG_sum2[0:LS//2]

SIG2_256= SIG2[0:LS//2]

SIG3_lin_302= SIG3_lin[0:LS3//2]

SIG4_256= SIG4[0:LS4//2]

SIG5_256= SIG5[0:LS5//2]

SIG6_256= SIG6[0:LS6//2]


########### FOURIERE ############################

FTSIG = 2*abs(np.fft.fft(SIG))**2 
FTSIG_CH = 2*abs(np.fft.fft(SIG*signal.chebwin(LS, at=100)))**2 
FTSIG_256 = 2*abs(np.fft.fft(SIG_256))**2 
FTSIG_CH_256 = 2*abs(np.fft.fft(SIG_256*signal.chebwin(LS//2, at=100)))**2 

FTSIG2 = 2*abs(np.fft.fft(SIG2))**2 
FTSIG2_CH = 2*abs(np.fft.fft(SIG2*signal.chebwin(LS, at=100)))**2 
FTSIG2_256 = 2*abs(np.fft.fft(SIG2_256))**2 
FTSIG2_CH_256 = 2*abs(np.fft.fft(SIG2_256*signal.chebwin(LS//2, at=100)))**2 

FTSIG3_lin = 2*abs(np.fft.fft(SIG3_lin))**2 
FTSIG3_lin_CH = 2*abs(np.fft.fft(SIG3_lin*signal.chebwin(LS3, at=100)))**2 
FTSIG3_lin_302 = 2*abs(np.fft.fft(SIG3_lin_302))**2 
FTSIG3_lin_CH_302 = 2*abs(np.fft.fft(SIG3_lin_302*signal.chebwin(LS3//2, at=100)))**2 

FTSIG4 = 2*abs(np.fft.fft(SIG4))**2 

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

##############  LINEARNA NAPOVED  ###########################################################################


def S_napov(ak, signal):
    napov=signal[0:len(signal)//2]
    N=len(napov)
    p=len(ak)
    for n in range(int(len(signal)/2)):
        napov=np.append(napov,sum(-ak[i]*napov[n+N-i-1] for i in range(p)))
    return napov
    
###########################################################################

F150=plt.figure(10)
F150=plt.subplot(1,1, 1 ) 
plt.title('Poli za vzorec  '+val[0]+' + šum2 ;  N='+str(LS),fontsize=16)
c=0
for i in [16,32,64]: # števila polov
#    ali2=a(SIG_256,i)
    ali2=a(SIG_noise1_256,i)
#    ali2=a(SIG_noise2_256,i)

    zk2=z(ali2)
    zk2_in=z_in(zk2)
    plt.scatter(zk2.real,zk2.imag, color=barva[c],alpha = 0.95,  marker='x', label='p={}'.format(i))
    plt.scatter(zk2_in.real,zk2_in.imag, color=barva[c],alpha = 0.5,   marker='s',label='p={}'.format(i))
    c=c+1
circle1 = plt.Circle((0, 0), 1, color=barva[0],fill=False)
F150.add_artist(circle1)

plt.xlabel('Re[z]',fontsize=16)
plt.ylabel('Im[z]',fontsize=16)
plt.legend(loc='best')

F50=plt.figure(50)

c=0
#    ali2=a(SIG_256,i)
for i in [16,32,64]: # števila polov
    ali2=a(SIG_noise1_256,i)
#    ali2=a(SIG_noise2_256,i)
    zk2=z(ali2)  ### brez normiranja
    
    napoved=S_napov(ali2,SIG) ### brez normiranja
    napoved=S_napov(ali2,SIG_sum1) ### brez normiranja
#    napoved=S_napov(ali2,SIG_sum2) ### brez normiranja
    N=len(napoved)
    F50=plt.subplot(2,2, 1 ) 
    plt.plot(T[LS//2:LS],napoved[LS//2:LS],color=barva[c],alpha = 0.6,label='p={}'.format(i))
    F50=plt.subplot(2,2, 2 ) 
#    plt.plot(T[LS//2:LS],SIG[LS//2:LS]-napoved[LS//2:LS],color=barva[c],ls='-',alpha = 0.75, label='p={}'.format(i))#+'{:.{}f}'.format(SHfi, 3 ))
    plt.plot(T[LS//2:LS],SIG_sum1[LS//2:LS]-napoved[LS//2:LS],color=barva[c],ls='-',alpha = 0.75, label='p={}'.format(i))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T[LS//2:LS],SIG_sum2[LS//2:LS]-napoved[LS//2:LS],color=barva[c],ls='-',alpha = 0.75, label='p={}'.format(i))#+'{:.{}f}'.format(SHfi, 3 ))
    c=c+1

F50=plt.subplot(2,2, 1 ) 
plt.title('Linearna napoved (brez normiranja)',fontsize=16)   
#plt.plot(T[LS//2:LS],SIG[LS//2:LS],color='k',ls='-',alpha = 0.4, label=val[0])#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T[LS//2:LS],SIG_sum1[LS//2:LS],color='k',ls='-',alpha = 0.4, label=val[0]+' + šum2')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[LS//2:LS],SIG_sum2[LS//2:LS],color='k',ls='-',alpha = 0.4, label=val[0])#+'{:.{}f}'.format(SHfi, 3 ))
plt.yscale('log')
plt.xlabel('t',fontsize=16)   
plt.ylabel('Amplituda',fontsize=16)   
plt.xlim([0.5,1])
plt.legend(loc=2)

F50=plt.subplot(2,2, 2 ) 
plt.title('Odstopanje linearne napovedi od pravega signala (brez normiranja)',fontsize=16)   
#plt.plot(T[LS//2:LS],SIG[LS//2:LS]-napoved[LS//2:LS],color='k',ls='-',alpha = 0.5, label=val[0])#+'{:.{}f}'.format(SHfi, 3 ))
plt.yscale('log')
plt.xlabel('t',fontsize=16)   
plt.ylabel(r'$|S(t)-\widetilde{S}(t)|$',fontsize=16)   
plt.xlim([0.5,1])
plt.legend(loc=2)

c=0
for i in [16,32,64]: # števila polov
#    ali2=a(SIG_256,i)
    ali2=a(SIG_noise1_256,i)
#    ali2=a(SIG_noise2_256,i)    
    zk2=z(ali2)
    alin2=ak_abs(zk2,ali2)  ###  normirano
    
#    napoved=S_napov(alin2,SIG) ###  normirano
    napoved=S_napov(alin2,SIG_sum1) ###  normirano
#    napoved=S_napov(alin2,SIG_sum2) ###  normirano    N=len(napoved)
    F50=plt.subplot(2,2, 3 ) 
    plt.plot(T[LS//2:LS],napoved[LS//2:LS],color=barva[c],alpha = 0.6,label='p={}'.format(i))
    F50=plt.subplot(2,2, 4 ) 
#    plt.plot(T[LS//2:LS],SIG[LS//2:LS]-napoved[LS//2:LS],color=barva[c],ls='-',alpha = 0.75, label='p={}'.format(i))#+'{:.{}f}'.format(SHfi, 3 ))
    plt.plot(T[LS//2:LS],SIG_sum1[LS//2:LS]-napoved[LS//2:LS],color=barva[c],ls='-',alpha = 0.75, label='p={}'.format(i))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T[LS//2:LS],SIG_sum2[LS//2:LS]-napoved[LS//2:LS],color=barva[c],ls='-',alpha = 0.75, label='p={}'.format(i))#+'{:.{}f}'.format(SHfi, 3 ))
    c=c+1

F50=plt.subplot(2,2, 3 ) 
plt.title('Linearna napoved (normirano)',fontsize=16)   
#plt.plot(T[LS//2:LS],SIG[LS//2:LS],color='k',ls='-',alpha = 0.4, label=val[0])#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T[LS//2:LS],SIG_sum1[LS//2:LS],color='k',ls='-',alpha = 0.4, label=val[0]+' + šum2')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T[LS//2:LS],SIG_sum2[LS//2:LS],color='k',ls='-',alpha = 0.4, label=val[0])#+'{:.{}f}'.format(SHfi, 3 ))
#plt.yscale('symlog')
plt.xlabel('t',fontsize=16)   
plt.ylabel('Amplituda',fontsize=16)   
plt.xlim([0.5,1])
plt.legend(loc=2)

F50=plt.subplot(2,2, 4 ) 
plt.title('Odstopanje linearne napovedi od pravega signala (normirano)',fontsize=16)   
#plt.plot(T[LS//2:LS],SIG[LS//2:LS]-napoved[LS//2:LS],color='k',ls='-',alpha = 0.5, label=val[0])#+'{:.{}f}'.format(SHfi, 3 ))
plt.yscale('log')
plt.xlabel('t',fontsize=16)   
plt.ylabel(r'$|S(t)-\widetilde{S}(t)|$',fontsize=16)   
plt.xlim([0.5,1])
plt.legend(loc=2)