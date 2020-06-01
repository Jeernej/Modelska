# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:27:18 2019

@author: jernej
"""
import scipy as sc
from scipy import special
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import colors

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation


font = {
        'color':  'k',
        'size': 20,
        'verticalalignment': 'bottom'
        }
            
crta=['c','m','y','b','r','g','k']       

e=2.718281828459045    
pi=3.141592653589793 
e0=8.8541878128*10**(-12)


def Hdata(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins='auto',normed=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram (vrednost Hi,položajxi), polažaji sredine binov, število binov
     
     
def energija(zacetni,N): 
#    N=7   ## <-- !! popravi za vsak N
    en=0
    for i in range(0,N-1):
#        print('i='+str(i))
        for j in range(i+1,N):
#            en = en + np.sqrt( 2 - 2 * ( np.cos(zacetni[i])*np.cos(zacetni[j])  +  np.sin(zacetni[j])*np.sin(zacetni[i]) * np.sin(zacetni[i+N]) * np.sin(zacetni[j+N]) + np.sin(zacetni[j])*np.sin(zacetni[i]) * np.cos(zacetni[i+N]) * np.cos(zacetni[j+N]) ))**(-1)
            en = en + np.sqrt( 2 - 2 * ( np.cos(zacetni[i])*np.cos(zacetni[j])  +  np.sin(zacetni[j]) * np.sin(zacetni[i]) * np.sin(zacetni[i+N]) * np.sin(zacetni[j+N]) + np.sin(zacetni[j])*np.sin(zacetni[i]) * np.cos(zacetni[i+N]) * np.cos(zacetni[j+N]) ))**(-1)
#            print('j='+str(j))
#            print('en='+str(en))
    return en
    
    
def Izrac_tezisca(ZacetniAnim): 
    N=ZacetniAnim.shape[1]/2
    k=ZacetniAnim.shape[0]
    TEZ_X=np.empty(0)
    TEZ_Y=np.empty(0)    
    TEZ_Z=np.empty(0)    
    for i in range(0,k):
        polozaj=ZacetniAnim[i]   
        TH=polozaj[0:N]
        FI=polozaj[N:2*N]    
        xN=np.sin(TH)*np.cos(FI)
        yN=np.sin(TH)*np.sin(FI)
        zN=np.cos(TH) 
        TEZ_X=np.append(TEZ_X, sum(xN)/N)
        TEZ_Y=np.append(TEZ_Y, sum(yN)/N) 
        TEZ_Z=np.append(TEZ_Z, sum(zN)/N)
    return TEZ_X,TEZ_Y,TEZ_Z
    
    
    
def Izrac_momenta(ZacetniAnim,l,m): 
    
    N=ZacetniAnim.shape[1]/2
    k=ZacetniAnim.shape[0]

    MOM=np.empty(0)    

    for i in range(0,k):
        polozaj=ZacetniAnim[i]
    
        Th=polozaj[0:N]
        Fi=polozaj[N:2*N]
        MOM=np.append( MOM, sum(  np.real( sc.special.sph_harm(m,l, Fi, Th) ) )/ N ) ## izračunam vsoto za vse položaje vseh n delcev in normiram z N
    return MOM
    

def Izrac_povprečja(vektor): 
    AVG=np.empty(0)    
    for i in range(1,len(vektor)):
        AVG=np.append( AVG, np.mean(vektor[0:i]) )      
    return AVG  

def Izrac_povprečja10(vektor): 
    AVG=np.empty(0) 
    for i in range(1,len(vektor)):
        AVG=np.append( AVG, np.mean(vektor[0:i]) )      
    return AVG  
    
    
###### ___zaćetne vrednosti

NN=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,40])
wiki=np.array([0.500000000,1.732050808,3.674234614,6.474691495,9.985281374,14.452977414,19.675287861,25.759986531,32.716949460,40.596450510,49.165253058,58.853230612,69.306363297,80.670244114,92.911655302,106.050404829,120.084467447,135.089467557,150.881568334, 223.347074052,660.675278835])


N=7
n=10**(4)
sirina=np.sqrt(8*pi/N/np.sqrt(3))
#sirina=sirina/10


th=np.random.rand(N)*pi
fi=np.random.rand(N)*pi*2
#zacetni=[th,fi]    ## zapakirani argumenti v eno spremenljivko
zacetni=np.append(th,fi)

###### ___minimizacija

Rl_Pot_En=energija(zacetni,N)
print(Rl_Pot_En)

#    
#for zz in range(1,6) :

temp =10**(-15)
I=np.empty(0)
SIRINA=np.empty(0)
RANDOM=np.empty(0)

R=np.empty(0)
E=np.empty(0)
MM=np.empty(0)
dE=np.empty(0)

SuS=np.empty(0)
Cv=np.empty(0)

TEMP=np.empty(0)

dTh=np.empty(0)
dFi=np.empty(0)

ZacetniAnim=np.ones(len(zacetni))*zacetni

#    for k in range(0,10):
l=0
k=0
on=0
for i in range(0,n):
	
    Nrand = np.random.randint(1,N) ## en elektron fiksen, da ni preveč vrtenja naokrog ## random integers from low (inclusive) to high (exclusive).
    th_rand = np.random.normal(0, sirina)
    fi_rand = np.random.normal(0, sirina)

    zacetni2=np.ones(2*N)*zacetni
    zacetni2[Nrand]=zacetni2[Nrand]+th_rand
    zacetni2[N+Nrand]=zacetni2[N+Nrand]+fi_rand
    
    # računanje dE 
    energ=energija(zacetni,N)
    deltaE = energija(zacetni2,N) - energ
  	
    if deltaE <= 0: 
        
        E=np.append(E, energ)
        dE=np.append(dE,deltaE)
        dTh=np.append(dTh,th_rand)
        dFi=np.append(dFi,fi_rand)
        
        k=k+1
        zacetni =  zacetni2
        ZacetniAnim=np.row_stack((ZacetniAnim,zacetni2))
#
#
        if abs(energ-wiki[N-2])<=0.1 and on==0:
            sirina=sirina/10
            on=1
            SIRINA=np.append(SIRINA, i+1)
            print('on='+str(on))

        if abs(energ-wiki[N-2])<=0.01 and on==1:
            sirina=sirina/10
            on=2
            SIRINA=np.append(SIRINA, i+1)
            print('on='+str(on))

        if abs(energ-wiki[N-2])<=0.001 and on==2:
            sirina=sirina/10
            on=3
            SIRINA=np.append(SIRINA, i+1)
            print('on='+str(on))
#
        if abs(energ-wiki[N-2])<=0.0001 and on==3:
            sirina=sirina/10
            on=4
            SIRINA=np.append(SIRINA, i+1)
            print('on='+str(on))

        if abs(energ-wiki[N-2])<=0.00001 and on==4:
            sirina=sirina/10
            on=5
            SIRINA=np.append(SIRINA, i+1)

#            print('on='+str(on))
#        if abs(energ-wiki[N-2])<=0.0000001 and on==5:
#            sirina=sirina/10
#            on=6
#            SIRINA=np.append(SIRINA, i+1)
#            print('on='+str(on))
  
        I=np.append(I,i+1) # vse izvedene poteze
        
#    elif np.random.random() <= np.exp(-deltaE/temp): 
#        zacetni =  zacetni2
#        RANDOM=np.append(RANDOM, i+1)
#        I=np.append(I,i+1) # vse izvedene poteze
#
##
#        E=np.append(E, energ)
#        dE=np.append(dE,deltaE)
#        dTh=np.append(dTh,th_rand)
#        dFi=np.append(dFi,fi_rand)
#
#        ZacetniAnim=np.row_stack((ZacetniAnim,zacetni2))
#
#        l=l+1

    R=np.append(R,(k+l)/(i+1)) # razmerje med izvedenimi in sprejetimi potezami
    
I=np.append(I,n) # vse izvedene poteze

SIRINA=np.append(SIRINA, 0)
RANDOM=np.append(RANDOM, 0)


#print('izračun energije po metropolisu za Ne='+str(N)+' pri Temp='+str(temp)+':')
print('izračun energije po metropolisu za Ne='+str(N)+':')
print(energija(zacetni,N))         
print('wiki='+str(wiki[N-2]))         
print(r'dE='+str( abs( wiki[N-2]-E[k-1]) ) )
print('k='+str(k))         
print('l='+str(l))         

Hfi, xfi, binEdge_fi, Lfi=Hdata(dFi)
Hth, xth, binEdge_th, Lth=Hdata(dTh)


TEZISC_x,TEZISC_y,TEZISC_z=Izrac_tezisca(ZacetniAnim)

#### vsota kvadratov m-komponent za vsak el je invariantna mera, 
Y_0_0=Izrac_momenta(ZacetniAnim,0,0) 

Y_1_0=Izrac_momenta(ZacetniAnim,1,0) 
Y_1_1=Izrac_momenta(ZacetniAnim,1,1) 
mY_1_1=Izrac_momenta(ZacetniAnim,1,-1) 
Y_1=Y_1_0**2+Y_1_1**2       +mY_1_1**2

Y_2_0=Izrac_momenta(ZacetniAnim,2,0) 
Y_2_1=Izrac_momenta(ZacetniAnim,2,1) 
Y_2_2=Izrac_momenta(ZacetniAnim,2,2)
mY_2_1=Izrac_momenta(ZacetniAnim,2,-1) 
mY_2_2=Izrac_momenta(ZacetniAnim,2,-2)
Y_2= Y_2_0**2+Y_2_1**2+Y_2_2**2     +mY_2_1**2+mY_2_2**2

Y_3_0=Izrac_momenta(ZacetniAnim,3,0) 
Y_3_1=Izrac_momenta(ZacetniAnim,3,1) 
Y_3_2=Izrac_momenta(ZacetniAnim,3,2) 
Y_3_3=Izrac_momenta(ZacetniAnim,3,3) 
mY_3_1=Izrac_momenta(ZacetniAnim,3,-1) 
mY_3_2=Izrac_momenta(ZacetniAnim,3,-2) 
mY_3_3=Izrac_momenta(ZacetniAnim,3,-3) 
Y_3=Y_3_0**2+Y_3_1**2+Y_3_2**2+Y_3_3**2     +mY_3_1**2+mY_3_2**2+mY_3_3**2

Y_4_0=Izrac_momenta(ZacetniAnim,4,0) 
Y_4_1=Izrac_momenta(ZacetniAnim,4,1) 
Y_4_2=Izrac_momenta(ZacetniAnim,4,2) 
Y_4_3=Izrac_momenta(ZacetniAnim,4,3) 
Y_4_4=Izrac_momenta(ZacetniAnim,4,4) 
mY_4_1=Izrac_momenta(ZacetniAnim,4,-1) 
mY_4_2=Izrac_momenta(ZacetniAnim,4,-2) 
mY_4_3=Izrac_momenta(ZacetniAnim,4,-3) 
mY_4_4=Izrac_momenta(ZacetniAnim,4,-4)
Y_4=Y_4_0**2+Y_4_1**2+Y_4_2**2+Y_4_3**2+Y_4_4**2    +mY_4_1**2+mY_4_2**2+mY_4_3**2+mY_4_4**2

Y_5_0=Izrac_momenta(ZacetniAnim,5,0) 
Y_5_1=Izrac_momenta(ZacetniAnim,5,1) 
Y_5_2=Izrac_momenta(ZacetniAnim,5,2) 
Y_5_3=Izrac_momenta(ZacetniAnim,5,3) 
Y_5_4=Izrac_momenta(ZacetniAnim,5,4) 
Y_5_5=Izrac_momenta(ZacetniAnim,5,5) 
mY_5_1=Izrac_momenta(ZacetniAnim,5,-1) 
mY_5_2=Izrac_momenta(ZacetniAnim,5,-2) 
mY_5_3=Izrac_momenta(ZacetniAnim,5,-3) 
mY_5_4=Izrac_momenta(ZacetniAnim,5,-4) 
mY_5_5=Izrac_momenta(ZacetniAnim,5,-5) 
Y_5=Y_5_0**2+Y_5_1**2+Y_5_2**2+Y_5_3**2+Y_5_4**2+Y_5_5**2   +mY_5_1**2+mY_5_2**2+mY_5_3**2+mY_5_4**2+mY_5_5**2


Y_6_0=Izrac_momenta(ZacetniAnim,6,0) 
Y_6_1=Izrac_momenta(ZacetniAnim,6,1) 
Y_6_2=Izrac_momenta(ZacetniAnim,6,2) 
Y_6_3=Izrac_momenta(ZacetniAnim,6,3) 
Y_6_4=Izrac_momenta(ZacetniAnim,6,4) 
Y_6_5=Izrac_momenta(ZacetniAnim,6,5) 
Y_6_6=Izrac_momenta(ZacetniAnim,6,6)
mY_6_1=Izrac_momenta(ZacetniAnim,6,-1) 
mY_6_2=Izrac_momenta(ZacetniAnim,6,-2) 
mY_6_3=Izrac_momenta(ZacetniAnim,6,-3) 
mY_6_4=Izrac_momenta(ZacetniAnim,6,-4) 
mY_6_5=Izrac_momenta(ZacetniAnim,6,-5) 
mY_6_6=Izrac_momenta(ZacetniAnim,6,-6) 
Y_6=Y_6_0**2+Y_6_1**2+Y_6_2**2+Y_6_3**2+Y_6_4**2+Y_6_5**2+Y_6_6**2    +mY_6_1**2+mY_6_2**2+mY_6_3**2+mY_6_4**2+mY_6_5**2+mY_6_6**2

avg_Y_0=Izrac_povprečja(Y_0_0)
avg_Y_1=Izrac_povprečja(Y_1)
avg_Y_2=Izrac_povprečja(Y_2)
avg_Y_3=Izrac_povprečja(Y_3)
avg_Y_4=Izrac_povprečja(Y_4)
avg_Y_5=Izrac_povprečja(Y_5)
avg_Y_6=Izrac_povprečja(Y_6)


#Izrac_povprečja(vektor)
avgE=Izrac_povprečja(E)
avgEE=Izrac_povprečja(E**2)
avgE2=Izrac_povprečja(E)**2

avgEE_avgE2=avgEE-avgE2


##################################### izris grafov ################################################

F100=plt.figure(100)

plt.suptitle(r'Normirani histogrami porazdelitve sprejetih korakov položaja za '+'{:.{}e}'.format(n, 0 )+' potez',fontsize=18)
F100=plt.subplot(1, 2, 1 ) 
plt.step(xfi,Hfi,'y',alpha = 0.5)
plt.xlabel(r'd$\phi$')
plt.ylabel('n')
#plt.legend(loc='best')
plt.title(r'porazdelitev $\phi$ korakov ')#+'(N='+str(N)+',M='+str(M)+')')

F100=plt.subplot(1, 2, 2 ) 
plt.step(xth,Hth,'r',alpha = 0.5)
plt.xlabel(r'd$\theta$')
plt.ylabel('n')
#plt.legend(loc='best')
plt.title(r'porazdelitev $\theta$ korakov ')#+'(N='+str(N)+',M='+str(M)+')')
#plt.tight_layout()

#
#
#F5=plt.figure(5)
#
#F5=plt.subplot(2, 1, 1 )
#plt.title(r'Potek kvadratnih odmikov energije pri konstantni temperaturi za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(avgEE_avgE2,'r',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'$\left< V_N^2 \right>-\left< V_N\right>^2$' ,fontsize=16)   
#plt.legend(loc=0,fontsize=16)
#
#F5=plt.subplot(2, 1, 2 )
#plt.title(r'Potek vsote kvadratov m-komponent momentov $\left<Y_l^m\right>$ za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(avg_Y_0,'k',alpha = 0.95,label=r'$\left<\sum_mY_{0m}^2\right>$='+'{:.{}e}'.format(avg_Y_0[-1], 1 ))
#plt.plot(avg_Y_1,'r',alpha = 0.95,label=r'$\left<\sum_mY_{1m}^2\right>$='+'{:.{}e}'.format(avg_Y_1[-1], 1 ))
#plt.plot(avg_Y_2,'b',alpha = 0.95,label=r'$\left<\sum_mY_{2m}^2\right>$='+'{:.{}e}'.format(avg_Y_2[-1], 1 ))
#plt.plot(avg_Y_3,'g',alpha = 0.95,label=r'$\left<\sum_mY_{3m}^2\right>$='+'{:.{}e}'.format(avg_Y_3[-1], 1 ))
#plt.plot(avg_Y_4,'y',alpha = 0.95,label=r'$\left<\sum_mY_{4m}^2\right>$='+'{:.{}e}'.format(avg_Y_4[-1], 1 ))
#plt.plot(avg_Y_5,'m',alpha = 0.95,label=r'$\left<\sum_mY_{5m}^2\right>$='+'{:.{}e}'.format(avg_Y_5[-1], 1 ))
#plt.plot(avg_Y_6,'c',alpha = 0.95,label=r'$\left<\sum_mY_{6m}^2\right>$='+'{:.{}e}'.format(avg_Y_6[-1], 1 ))
#plt.axhline(y=0, color='k', linestyle='--')
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'vrednost momenta' ,fontsize=16)   
#plt.legend(loc=0,fontsize=16)
##plt.tight_layout()
#



F50=plt.figure(50)

F50=plt.subplot(3, 1, 1 )
#plt.title(r'Potek energije za '+str(k)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in Temp='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
plt.title(r'Potek energije in njenih  kvadratnih odmikov za '+str(k)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N),fontsize=18)#+'(N='+str(N)+',M='+str(M)+')')
plt.plot(E,'r',alpha = 0.95,label=r'$V_N$='+'{:.{}f}'.format(E[-1], 9 )+' ; izračun')#+'{:.{}f}'.format(SHfi, 3 ))
plt.axhline(y=wiki[N-2], color='k', linestyle='--',label=r'$V_N^{min}$='+str(wiki[N-2])+' ; Vir [2]')
plt.plot(avgEE_avgE2,'r:',alpha = 0.95,label=r'$\left< V_N^2 \right>-\left< V_N\right>^2$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xlabel(r'sprejete poteze' ,fontsize=18)   
plt.ylabel(r'energija' ,fontsize=18)   
plt.legend(loc=4,fontsize=18)

F50=plt.subplot(3, 1, 2 )
#plt.title(r'Spremembe energije za vsako od k='+str(k)+r' sprejetih potez pri $N_e$='+str(N)+' in Temp='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
plt.title(r'Spremembe energije za vsako od k='+str(k)+r' sprejetih potez pri $N_e$='+str(N),fontsize=18)#+'(N='+str(N)+',M='+str(M)+')')
plt.plot(abs(dE),'b',alpha = 0.95,label=r'sprejete $dV_N^{min}$ ')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(abs(E-np.ones(k+l)*wiki[N-2]),'k',alpha = 0.95,label=r'odstopanje $V_N^{min}$ od Vir [2]')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xlabel(r'sprejete poteze' ,fontsize=18)   
plt.ylabel(r'|d$V_N^{min}$|' ,fontsize=18)     
#plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best',fontsize=18)

#F50=plt.subplot(3, 1, 3 )
#plt.title(r'Potek kvadratnih odmikov energije pri konstantni temperaturi za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(avgEE_avgE2,'r',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'$\left< V_N^2 \right>-\left< V_N\right>^2$' ,fontsize=16)   
#plt.legend(loc=0,fontsize=16)
##

F50=plt.subplot(3, 1, 3 )
plt.title(r'Sprejetost potez tekom izvajanja potez $i$=1,..,n',fontsize=18)#+'(N='+str(N)+',M='+str(M)+')')
plt.plot(R,'r',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
preklop=0
randPreklop=0
for i in I: 
    if int(i)==int(SIRINA[preklop]):
        preklop=preklop+1
        plt.axvline(x=i,color='b')#+'{:.{}f}'.format(SHfi, 3 ))
#    if int(i)==int(RANDOM[randPreklop]):
#        randPreklop=randPreklop+1
#        plt.axvline(x=i,color='g',alpha = 0.3)#+'{:.{}f}'.format(SHfi, 3 ))
#    else: plt.axvline(x=i,color='k',alpha = 0.05)#+'{:.{}f}'.format(SHfi, 3 ))
plt.xlabel(r'izvedene poteze' ,fontsize=18)   
plt.ylabel(r'R=k/i' ,fontsize=18)     
#plt.xscale('log')
#plt.yscale('log')
#plt.legend(loc='best',fontsize=16)
#plt.tight_layout()



F1000=plt.figure(1000)
F1000=plt.subplot(2, 1, 1 ) 
#plt.title(r'Potek težišča za '+str(k)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in Temp='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
plt.title(r'Potek težišča za '+str(k)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N),fontsize=18)#+'(N='+str(N)+',M='+str(M)+')')
plt.plot(TEZISC_x,'r:',alpha = 0.95,label=r'$x_T$='+'{:.{}e}'.format(TEZISC_x[-1], 1 ))
plt.plot(TEZISC_y,'g:',alpha = 0.95,label=r'$y_T$='+'{:.{}e}'.format(TEZISC_z[-1], 1 ))
plt.plot(TEZISC_z,'b:',alpha = 0.95,label=r'$z_T$='+'{:.{}e}'.format(TEZISC_y[-1], 1 ))
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel(r'sprejete poteze' ,fontsize=18)   
plt.ylabel(r'vrednost koordinate' ,fontsize=18)   
#plt.yscale('log')
plt.legend(loc='best',fontsize=18)

F1000=plt.subplot(2, 1, 2 )
##plt.title(r'Potek momentov $\left<Y_l^m\right>$ za '+str(k)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in Temp='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.title(r'Potek momentov $\left<Y_l^m\right>$ za '+str(k)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(Y_0_0,'k:',alpha = 0.95,label=r'$\left<Y_0^0\right>$='+'{:.{}e}'.format(Y_0_0[-1], 1 ))
#plt.plot(Y_1_0,'r:',alpha = 0.95,label=r'$\left<Y_1^0\right>$='+'{:.{}e}'.format(Y_1_0[-1], 1 ))
#plt.plot(Y_1_1,'b:',alpha = 0.95,label=r'$\left<Y_1^1\right>$='+'{:.{}e}'.format(Y_1_1[-1], 1 ))
#plt.plot(Y_2_0,'g:',alpha = 0.95,label=r'$\left<Y_2^0\right>$='+'{:.{}e}'.format(Y_2_0[-1], 1 ))
#plt.plot(Y_2_1,'y:',alpha = 0.95,label=r'$\left<Y_2^1\right>$='+'{:.{}e}'.format(Y_2_1[-1], 1 ))
#plt.plot(Y_2_2,'m:',alpha = 0.95,label=r'$\left<Y_2^2\right>$='+'{:.{}e}'.format(Y_2_2[-1], 1 ))
#plt.axhline(y=0, color='k', linestyle='--')
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'vrednost momenta' ,fontsize=16)   
##plt.yscale('log')
#plt.legend(loc='best',fontsize=16)
##plt.tight_layout()
plt.title(r'Potek vsote kvadratov m-komponent momentov $\left<Y_l^m\right>$ za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N),fontsize=18)#+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(avg_Y_0,'k',alpha = 0.95,label=r'$\left<\sum_mY_{0m}^2\right>$='+'{:.{}e}'.format(avg_Y_0[-1], 1 ))
#plt.plot(avg_Y_1,'r',alpha = 0.95,label=r'$\left<\sum_mY_{1m}^2\right>$='+'{:.{}e}'.format(avg_Y_1[-1], 1 ))
#plt.plot(avg_Y_2,'b',alpha = 0.95,label=r'$\left<\sum_mY_{2m}^2\right>$='+'{:.{}e}'.format(avg_Y_2[-1], 1 ))
#plt.plot(avg_Y_3,'g',alpha = 0.95,label=r'$\left<\sum_mY_{3m}^2\right>$='+'{:.{}e}'.format(avg_Y_3[-1], 1 ))
#plt.plot(avg_Y_4,'y',alpha = 0.95,label=r'$\left<\sum_mY_{4m}^2\right>$='+'{:.{}e}'.format(avg_Y_4[-1], 1 ))
#plt.plot(avg_Y_5,'m',alpha = 0.95,label=r'$\left<\sum_mY_{5m}^2\right>$='+'{:.{}e}'.format(avg_Y_5[-1], 1 ))
#plt.plot(avg_Y_6,'c',alpha = 0.95,label=r'$\left<\sum_mY_{6m}^2\right>$='+'{:.{}e}'.format(avg_Y_6[-1], 1 ))

plt.plot(Y_0_0**2,'k',alpha = 0.95,label=r'$\sum_mY_{0m}^2$='+'{:.{}e}'.format(Y_0_0[-1]**2, 1 ))
plt.plot(Y_1,'r',alpha = 0.95,label=r'$\sum_mY_{1m}^2$='+'{:.{}e}'.format(Y_1[-1], 1 ))
plt.plot(Y_2,'b',alpha = 0.95,label=r'$\sum_mY_{2m}^2$='+'{:.{}e}'.format(Y_2[-1], 1 ))
plt.plot(Y_3,'g',alpha = 0.95,label=r'$\sum_mY_{3m}^2$='+'{:.{}e}'.format(Y_3[-1], 1 ))
plt.plot(Y_4,'y',alpha = 0.95,label=r'$\sum_mY_{4m}^2$='+'{:.{}e}'.format(Y_4[-1], 1 ))
plt.plot(Y_5,'m',alpha = 0.95,label=r'$\sum_mY_{5m}^2$='+'{:.{}e}'.format(Y_5[-1], 1 ))
plt.plot(Y_6,'c',alpha = 0.95,label=r'$\sum_mY_{6m}^2$='+'{:.{}e}'.format(Y_6[-1], 1 ))

plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel(r'sprejete poteze' ,fontsize=18)   
plt.ylabel(r'vrednost momenta' ,fontsize=18)   
plt.legend(loc=1,fontsize=18)
#plt.tight_layout()



###### ____izris_sfer z naboji_______________________________

xN=np.sin(th)*np.cos(fi)
yN=np.sin(th)*np.sin(fi)
zN=np.cos(th)

F20=plt.figure(figsize=(10,12))
#F20=plt.subplot(1, 2, 1 ) 
Axes3D = plt.axes(projection='3d')
Axes3D.scatter(xN, yN, zN, zdir='z',marker='o', s=10, c='r', depthshade=True)
plt.title('Naključna začetna porazdelitev elektronov po enotski krogli ;   $N_e$='+str(N))

# draw sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)

plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
#plt.zlim([-1.2,1.2])
plt.tight_layout()




Th=zacetni[0:N]
Fi=zacetni[N:2*N]

XN=np.sin(Th)*np.cos(Fi)
YN=np.sin(Th)*np.sin(Fi)
ZN=np.cos(Th)

F10=plt.figure(figsize=(10,12))
#figsize=(10,6)
#F20=plt.subplot(1, 2, 2 ) 
Axes3D = plt.axes(projection='3d')
Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True)#,label=r'$V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[k+l-1])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[k+l-1]))+'\n d$V_N^k=$'+str(dE[k+l-1])+'\n izvedene poteze $n$='+str(I[k])+'\n sprejete poteze $k$='+str(k))
plt.title(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N))
#plt.title(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N)+'\n $V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[k+l-1])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[k+l-1]))+'\n d$V_N^k=$'+str(dE[k+l-1])+'\n izvedene poteze $n$='+str(int(I[k]))+'\n sprejete poteze $k$='+str(k))
#plt.legend(loc=3,fontsize=16)

# draw sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)

plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
#plt.zlim([-1.2,1.2])
plt.tight_layout()




#######################################################################################
####### izdelava animacije
###############################################


Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=360)

F110=plt.figure(figsize=(8,8))

Axes3D = plt.axes(projection='3d')
#plt.title(r'Porazdelitev elektronov po enotski krogli za minimizacijo energije ;   $N_e$='+str(N))
    
# draw sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)    
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.tight_layout()


def animate(i):

    polozaj=ZacetniAnim[i]
    
    Th=polozaj[0:N]
    Fi=polozaj[N:2*N]
    
    XN=np.sin(Th)*np.cos(Fi)
    YN=np.sin(Th)*np.sin(Fi)
    ZN=np.cos(Th)

    TEZ_X= sum(XN)/N
    TEZ_Y= sum(YN)/N
    TEZ_Z= sum(ZN)/N
    plt.suptitle(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N)+'\n $V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[i])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^k=$'+str(dE[i])+'\n izvedene poteze $n$='+str(int(I[i+1]))+'\n sprejete poteze $k$='+str(i+1))        
#    plt.suptitle(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N)+'T='+'{:.{}e}'.format(temp, 1 )+'\n $V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[i])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^k=$'+str(dE[i])+'\n izvedene poteze $n$='+str(int(I[i+1]))+'\n sprejete poteze $k$='+str(i+1))
    Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True)
    Axes3D.scatter(TEZ_X, TEZ_Y, TEZ_Z, zdir='z',marker='x', s=10, c='b', depthshade=True)

    Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True,label=r'$V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N=$'+str(E[i])+'\n $|V_N^i-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^i=$'+str(dE[i])+'sprejeta poteza i='+str(i)+ 'od izvedenih n='+str(I[i]))
    Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True,label=r'$V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[i])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^k=$'+str(dE[i])+'\n izvedene poteze $n$='+str(int(I[i+1]))+'\n sprejete poteze $k$='+str(i+1))
    plt.legend(loc=3,fontsize=16)
#
ani = matplotlib.animation.FuncAnimation(F110, animate, frames=k+l, repeat=False)
ani.save('sprejemanje_potez_N'+str(int(N))+'_b.mp4', writer=writer)
#

#
#
