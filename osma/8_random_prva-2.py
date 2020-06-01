# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
import matplotlib.pyplot  as plt
from scipy import stats
from scipy.optimize import curve_fit
import timeit



def BM(a1,a2): #Box-Muller
    pi=3.141592653589793 
    z1 = (-2 * np.log(a1))**(0.5) * np.sin(2 * pi * a2)
    z2 = (-2 * np.log(a1))**(0.5) * np.cos(2 * pi * a2)
    return z1,z2

def Hdata0(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins=10,density=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,L-1 #prebinan histogram, polažaji sredine binov, število binov

def Hdata(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins='auto',density=False)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,L-1 #prebinan histogram, polažaji sredine binov, število binov
    
def Fit0(dogodki,funkcija): #fitanje na vrednosti histogramov
    h,x,L=Hdata0(dogodki)  # generira normaliziran histogram
    p=[0,1] #začetni približek [mu,std]
    coeff, var_matrix = curve_fit(funkcija, x, h, p0=p)
    return coeff #[mu,std]
    
def Fit(dogodki,funkcija): #fitanje na vrednosti histogramov
    p= Fit0(dogodki,funkcija) # generira začetni približek [mu,std]
    h,x,La=Hdata(dogodki)   # generira  histogram
#    p=[-0.1,0.5] #začetni približek [mu,std]
    coeff, var_matrix = curve_fit(funkcija, x, h, p0=p)
    return coeff #[mu,std]
    
def Fdata(x,*p):  #funkcija za fitanje gaussove porazdelitve
    mu,std=p
    e=2.718281828459045    
    pi=3.141592653589793 
    F=e**(-(((x - mu)/std)**2.)/2)/(2.*pi)**(0.5)/std
    return F

      
def fdata(x,mu,std): #funkcija za računanje vrednosti gaussove porazdelitve v sredinah položajev binov
    e=2.718281828459045    
    pi=3.141592653589793 
    F=e**(-(((x - mu)/std)**2.)/2)/(2.*pi)**(0.5)/std
    return F
#____________________________________________________ 
# generiranje števil in statistična analiza



M=10000 #št. ponovitev za porazdelitev statistike

X2a1=np.empty(M) 
X2a=np.empty(M) 
X2z1=np.empty(M)
X2z2=np.empty(M)
X2K1=np.empty(M)

La1=np.empty(M) 
La0=np.empty(M) 
La=np.empty(M) 
Lz1=np.empty(M)
Lz2=np.empty(M)
LK1=np.empty(M)

KuDa1=np.empty(M)
KnDa=np.empty(M)
KnDz1=np.empty(M)
KnDz2=np.empty(M)
KnDK1=np.empty(M)

KDz12=np.empty(M)
KDz1a=np.empty(M)
KDz2a=np.empty(M)
KDK1a=np.empty(M)
    
i=0
while i<M:

    N=10000 # velikost vzorcev
    
    a0=np.random.randn(N)  # generiranje števil normal
    a=np.random.randn(N)  # generiranje števil normal

        
    a1=np.random.rand(N)  # generiranje števil uniform
    a2=np.random.rand(N)
    z1,z2=BM(a1,a2)       # generiranje števil normal B-M

    a3=np.random.rand(N) # generiranje števil uniform za konvolucijo K1
    a4=np.random.rand(N)
    a5=np.random.rand(N)
    a6=np.random.rand(N)
    a7=np.random.rand(N)
    a8=np.random.rand(N)
    a9=np.random.rand(N)
    a10=np.random.rand(N)
    a11=np.random.rand(N)
    a12=np.random.rand(N)
    K1=a1+a2+a3+a4+a5+a6-a7-a8-a9-a10-a11-a12  # generiranje števil normal konvolucijsko

    
    # priprava na chi2 statistična analiza

    Ha1,xa1,La1[i]=Hdata(a1)  # normaliziacija histograma za chi test 

    Ha0,xa0,La0[i]=Hdata(a0)  # normaliziacija histograma za fitanje
    Ha,xa,La[i]=Hdata(a)  # normaliziacija histograma za fitanje
    Hz1,xz1,Lz1[i]=Hdata(z1)
    Hz2,xz2,Lz2[i]=Hdata(z2)    
    HK1,xK1,LK1[i]=Hdata(K1)
 
    coeffa = Fit0(a,Fdata) # FITANJE norm na histogram za fitanje
    coeffz1 = Fit0(z1,Fdata)
    coeffz2 = Fit0(z2,Fdata)
    coeffK1 = Fit0(K1,Fdata)
  
#    coeffa = Fit(a,Fdata) # FITANJE norm na histogram za fitanje
#    coeffz1 = Fit(z1,Fdata)
#    coeffz2 = Fit(z2,Fdata)
#    coeffK1 = Fit(K1,Fdata)
#   
    Fa=fdata(xa,coeffa[0], coeffa[1])*N*(xa[1]-xa[0])  # računanje vrednosti FITA glede na stolpce histograma za fitanje
    Fz1=fdata(xz1,coeffz1[0], coeffz1[1])*N*(xz1[1]-xz1[0])
    Fz2=fdata(xz2,coeffz2[0], coeffz2[1])*N*(xz2[1]-xz2[0])
    FK1=fdata(xK1,coeffK1[0], coeffK1[1])*N*(xK1[1]-xK1[0])

    
    X2a[i]=stats.chisquare(Ha,Fa)[0]   # chi test za odtopanje vrednosti FITA norm glede na vrednosti normaliziranega histograma
    X2z1[i]=stats.chisquare(Hz1,Fz1)[0]  
    X2z2[i]=stats.chisquare(Hz2,Fz2)[0]
    X2K1[i]=stats.chisquare(HK1,FK1)[0]
    
    
    # chi2 statistična analiza
#    X2a1[i]=stats.chisquare(a1)[0] # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
    X2a1[i]=stats.chisquare(Ha1)[0] # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
#    X2a[i]=stats.chisquare(Ha)[0]   # chi test za odtopanje vrednosti FITA norm glede na vrednosti normaliziranega histograma
#    X2z1[i]=stats.chisquare(Hz1)[0]  
#    X2z2[i]=stats.chisquare(Hz2)[0]
#    X2K1[i]=stats.chisquare(HK1)[0]

#    X2a[i]=stats.chisquare(Ha,Ha0)[0]   # chi test za odtopanje vrednosti FITA norm glede na vrednosti normaliziranega histograma
#    X2z1[i]=stats.chisquare(Hz1,Ha0)[0]  
#    X2z2[i]=stats.chisquare(Hz2,Ha0)[0]
#    X2K1[i]=stats.chisquare(HK1,Ha0)[0]    
    
    # Kolmogorov - Smirnov  test
    KuDa1[i]=stats.kstest(a1, 'uniform')[0]  # K-S test against unirorm function
    KnDa[i]=stats.kstest(a, 'norm')[0]      # K-S test against normal function
    KnDz1[i]=stats.kstest(z1, 'norm')[0]    # K-S test against normal function
    KnDz2[i]=stats.kstest(z2, 'norm')[0]    # K-S test against normal function
    KnDK1[i]=stats.kstest(K1, 'norm')[0]    # K-S test against normal function
    
    KDz12[i]=stats.ks_2samp(z1,z2)[0] # K-S test between B-M generated normal numbers 
    KDz1a[i]=stats.ks_2samp(z1,a)[0]  # K-S test against normal generated a
    KDz2a[i]=stats.ks_2samp(z2,a)[0]  # K-S test against normal generated a
    KDK1a[i]=stats.ks_2samp(K1,a)[0]  # K-S test against normal generated a
    
    i=i+1 

#### Loto od 1991 dalje 

#### možne števlke 
XLOTO=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
#### frekvenca izžrebov vsake od možnih števlk
LOTO=np.array([405,400,377,435,408,392,420,417,397,382,444,441,414,399,414,429,404,423,378,419,412,409,397,388,421,408,422,403,377,403,391,421,400,383,421,418,409,418,397])
# chi2 statistična analiza
X2LOTO=stats.chisquare(LOTO) ## nenormalizirano = 26.564544539506795

#### generiranje vektorja izžrebanih števil glede na statistike
loto=np.empty(sum(LOTO))
g=0
d=0
while g<len(LOTO):
    f=0
    while f<LOTO[g]:
        
        loto[d]=(g+1)/39 ## delim z 39 da dobim vrednosti za možne števlke znotraj intervala (0,1]
        f=f+1
        d=d+1        
    g=g+1
    
##X2loto=stats.chisquare(loto)[0] # chi2 statistična analiza = 2577.6015798976637
KuDloto=stats.kstest(loto, 'uniform')[0] # K-S test against unirorm function = 0.028339656485101894

Hloto,xloto,Lloto=Hdata(loto)  # normaliziacija histograma za chi test 
X2HLOTO=stats.chisquare(Hloto)[0]  # chi2 statistična analiza NORMALIZIRANO = 0.0053631242868862946

###### ____izris_grafov_in_histogramov______________________________

F10=plt.figure(10)
F10=plt.subplot(2, 2, 1 )  
plt.step(xz1,Hz1,'c',label=r'$\tt{data\ z1}$')
plt.plot(xz1,Fz1,'k',label=r'$\tt{fit\ z1}$')
plt.plot(xz1,abs(Hz1-Fz1),'y',label=r'$\tt{|data-fit|}$')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(loc='best')
F10=plt.subplot(2, 2, 2 )  
plt.step(xz2,Hz2,'m',label=r'$\tt{data\ z2}$')
plt.plot(xz2,Fz2,'k',label=r'$\tt{fit\ z2}$')
plt.plot(xz2,abs(Hz2-Fz2),'y',label=r'$\tt{|data-fit|}$')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(loc='best')
F10=plt.subplot(2, 2, 3 )  
plt.step(xK1,HK1,'r',label=r'$\tt{data\ K1}$')
plt.plot(xK1,FK1,'k',label=r'$\tt{fit\ K1}$')
plt.plot(xK1,abs(HK1-FK1),'y',label=r'$\tt{|data-fit|}$')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(loc='best')
F10=plt.subplot(2, 2, 4 )  
#plt.hist(K1,bins='auto',normed=True,histtype='step')
plt.step(xa,Ha,'k',label=r'$\tt{data\ norm}$')
plt.plot(xa,Fa,'k',label=r'$\tt{fit\ norm}$')
plt.plot(xa,abs(Ha-Fa),'y',label=r'$\tt{|data-fit|}$')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(loc='best')
plt.suptitle('Normirani histogrami greneriranih normalnih porazdelitev  '+'(N='+str(N)+',M='+str(M)+')',fontsize=16)

F2=plt.figure(1)
#if N>10000:
#    Ax1 = F2.add_subplot(2,2,1)
#else:
F2=plt.subplot(2, 2, 1 )
plt.hist(a1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{a1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(a2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{a2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
if N==10000: 
    plt.hist(loto, bins='auto',normed=True,facecolor="None",edgecolor='g',label=r'$\tt{loto}$ (N='+str(sum(LOTO))+r',$\nu$='+str(len(LOTO)-1)+r',$\chi^2$='+'{:.{}f}'.format(X2LOTO[0], 3 )+r',$p$='+'{:.{}f}'.format(X2LOTO[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.title('Normirani histogrami enakomerne porazdelitve '+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel('x')
plt.ylabel('n')
#plt.ylabel('n')
plt.legend(loc=4)
#if N>10000:
#    Ax2 = F2.add_subplot(2,2,1, sharex=Ax1, frameon=False)
#    plt.step(XLOTO/len(XLOTO),LOTO/np.mean(LOTO),color='g',label=r'$\tt{LOTO}$ '+r'($\chi^2$='+'{:.{}f}'.format(X2loto, 3 )+')',alpha = 0.5,where='pre')#,ls='step')  # arguments are passed to np.histogram
#    plt.step(CENKE/len(XLOTO),ENKE,color='k',label=r'$\tt{uniform}$ '+r'($\chi^2$='+'{:.{}f}'.format(X2enke, 3 )+')',alpha = 0.5,where='pre')#,ls='step')  # arguments are passed to np.histogram    
#    Ax2.yaxis.tick_right()
#    Ax2.yaxis.set_label_position("right")
#    Ax2.set_ylabel(r'$\tt{LOTO}$', color='g')
#    Ax2.set_ylim([0,1.2])
#    Ax2.tick_params('y', colors='g')
#    plt.legend(loc=4)

#if N=10000:
#    aAx1 = F2.add_subplot(2,2,2)
#else:
F2=plt.subplot(2, 2, 2 )  
plt.hist(a1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r'$\tt{a1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(a2, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',label=r'$\tt{a2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
if N==10000:
    plt.hist(loto, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='g',label=r'$\tt{loto}$ (N='+str(sum(LOTO))+r',D='+'{:.{}f}'.format(KuDloto, 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.title('Kumilativni normirani histogrami enakomerne porazdelitve  '+'(N='+str(N)+',M='+str(M)+')')
plt.ylabel('kumulativen(n)')
plt.xlabel('x')
plt.legend(loc=4)
#if N==10000: 
#    aAx2 = F2.add_subplot(2,2,2, sharex=aAx1, frameon=False)
#    plt.step(XLOTO/len(XLOTO),CLOTO/np.sum(LOTO),color='g',label=r'$\tt{LOTO}$ '+'(D='+'{:.{}f}'.format(KuDloto, 3 )+')',alpha = 0.5,where='pre')#,ls='step')  # arguments are passed to np.histogram
#    aAx2.yaxis.tick_right()
#    aAx2.yaxis.set_label_position("right")
#    aAx2.set_ylabel(r'$\tt{LOTO}$', color='g')
#    aAx2.set_ylim([0,1.2])
#    aAx2.tick_params('y', colors='g')
#    plt.legend(loc=4)

F2=plt.subplot(2, 2, 3 )
plt.hist(z1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{z1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(z2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{z2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(K1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{K1}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(a, bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{norm}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Normirani histogrami normalne porazdelitve  '+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(loc='best')
plt.show()
F2=plt.subplot(2, 2, 4 )
plt.hist(z1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r'$\tt{z1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(z2, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',label=r'$\tt{z2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(K1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='r',label=r'$\tt{K1}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(a, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='k',label=r'$\tt{norm}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Kumilativni normirani histogrami normalne porazdelitve  '+'(N='+str(N)+',M='+str(M)+')')
plt.ylabel('kumulativen(n)')
plt.xlabel('x')
plt.legend(loc='best')
plt.show()

#F3=plt.figure(3)
#plt.hist(X2a,bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{chisquare(a) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
#plt.hist(X2a1,bins='auto',normed=True,facecolor="None",edgecolor='g',label=r'$\tt{chisquare(a1) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
#plt.hist(X2z1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{chisquare(z1) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.hist(X2z2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{chisquare(z2) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
#plt.hist(X2K1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{chisquare(K1) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.title('Histogrami porazdelitev vrednosti '+r'$\tt{\chi^2}$')
#plt.legend(loc=1)
#plt.show()

F3=plt.figure(3)
ax1 = F3.add_subplot(211)
#line1 = ax1.plot([1,3,4,5,2], 'o-')
plt.hist(X2a,  bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{chisquare(a,fit(a)) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
plt.hist(X2z1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{chisquare(z1.fit(z1)) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(X2z2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{chisquare(z2,fit(z2)) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
plt.hist(X2K1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{chisquare(K1,fit(K1)) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Normirani  histogrami porazdelitev vrednosti '+r'$\tt{\chi^2}$' +'(N='+str(N)+',M='+str(M)+')')
ax1.set_xlabel(r'$\tt{\chi^2}$')
ax1.set_xlim([0,2*len(xa)])
ax1.set_ylabel('n')
#plt.ylabel("Left Y-Axis Data")
plt.legend(loc=1)
# now, the second axes that shares the x-axis with the ax1
ax2 = F3.add_subplot(211, sharex=ax1, frameon=False)
plt.hist(X2a1,bins='auto',normed=True,facecolor="None",edgecolor='g',label=r"$\tt{chisquare(a1,'uniform') }$",alpha = 0.4,histtype='step') # arguments are passed to np.histogram
#line2 = ax2.plot([10,40,20,30,50], 'xr-')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_xlim([0,2*len(xa)])
ax2.set_ylabel(r'n iz $\tt{chisquare(a1) }$ podatkov', color='g')
ax2.tick_params('y', colors='g')
plt.legend(loc=4)
plt.show()
F3.add_subplot(212)
plt.hist(La-1, bins='auto',normed=True,histtype='step',color='k',alpha = 0.5,label=r"$\tt{\nu_{a}}$")  # arguments are passed to np.histogram
plt.hist(Lz1-1, bins='auto',normed=True,histtype='step',color='c',alpha = 0.5,label=r"$\tt{\nu_{z1}}$")  # arguments are passed to np.histogram
plt.hist(Lz2-1, bins='auto',normed=True,histtype='step',color='m',alpha = 0.5,label=r"$\tt{\nu_{z2}}$")  # arguments are passed to np.histogram
plt.hist(LK1-1, bins='auto',normed=True,histtype='step',color='r',alpha = 0.5,label=r"$\tt{\nu_{K1}}$")  # arguments are passed to np.histogram
plt.hist(La1-1, bins='auto',normed=True,histtype='step',color='g',alpha = 0.5,label=r"$\tt{\nu_{a1}}$")  # arguments are passed to np.histogram
plt.title(r'Prostostne stopnje $\nu$ histogramov iz katerih so računani '+r'$\tt{\chi^2}$ '+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel(r'$\nu$')
plt.xlim([0,2*len(xa)])
plt.ylabel('n')
plt.legend(loc=1)
plt.show()


F4=plt.figure(4)
F4=plt.subplot(2, 1, 1 )
plt.hist(KDz12*np.sqrt(N), bins='auto',normed=True,histtype='step',color='y',alpha = 0.5,label=r"$\tt{ks\_2samp(z1,z2)}$")  # arguments are passed to np.histogram
plt.hist(KDz1a*np.sqrt(N), bins='auto',normed=True,histtype='step',color='c',alpha = 0.5,label=r"$\tt{ks\_2samp(z1,a)}$")  # arguments are passed to np.histogram
plt.hist(KDz2a*np.sqrt(N), bins='auto',normed=True,histtype='step',color='m',alpha = 0.5,label=r"$\tt{ks\_2samp(z2,a)}$")  # arguments are passed to np.histogram
plt.hist(KDK1a*np.sqrt(N), bins='auto',normed=True,histtype='step',color='r',alpha = 0.5,label=r"$\tt{ks\_2samp(K1,a)}$")  # arguments are passed to np.histogram
plt.title(r'Normirani histogrami porazdelitev vrednosti D$\sqrt{N}$ primerjalnega K-S testa  '+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel(r'D$\sqrt{N}$')
plt.ylabel('n')
plt.legend(loc=1)
plt.show()
F4=plt.subplot(2, 1, 2 )
plt.hist(KnDa*np.sqrt(N), bins='auto',normed=True,histtype='step',color='k',alpha = 0.4,label=r"$\tt{kstest(a, 'norm')}$")  # arguments are passed to np.histogram
plt.hist(KnDz1*np.sqrt(N), bins='auto',normed=True,histtype='step',color='c',alpha = 0.4,label=r"$\tt{kstest(z1, 'norm')}$")  # arguments are passed to np.histogram
plt.hist(KnDz2*np.sqrt(N), bins='auto',normed=True,histtype='step',color='m',alpha = 0.4,label=r"$\tt{kstest(z2, 'norm')}$")  # arguments are passed to np.histogram
plt.hist(KnDK1*np.sqrt(N), bins='auto',normed=True,histtype='step',color='r',alpha = 0.4,label=r"$\tt{kstest(K1, 'norm')}$")  # arguments are passed to np.histogram
plt.hist(KuDa1*np.sqrt(N), bins='auto',normed=True,histtype='step',color='b',alpha = 0.5,label=r"$\tt{kstest(a1, 'uniform')}$")  # arguments are passed to np.histogram
plt.title(r'Normirani histogrami porazdelitev vrednosti D$\sqrt{N}$ primerjalnega K-S testa  '+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel(r'D$\sqrt{N}$')
plt.ylabel('n')
plt.legend(loc=1)
plt.show()
