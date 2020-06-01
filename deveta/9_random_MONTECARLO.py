# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
import scipy as sc
import matplotlib.pyplot  as plt
from scipy import stats
from scipy.optimize import curve_fit

e=2.718281828459045    
pi=3.141592653589793 

def BM(a1,a2): #Box-Muller
    pi=3.141592653589793 
    z1 = (-2 * np.log(a1))**(0.5) * np.sin(2 * pi * a2)
    z2 = (-2 * np.log(a1))**(0.5) * np.cos(2 * pi * a2)
    return z1,z2

def Hdata(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins='auto',normed=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram, polažaji sredine binov, število binov
     
def Fdata(x,*p):  #funkcija za fitanje gaussove porazdelitve
    mu,std=p
    e=2.718281828459045    
    pi=3.141592653589793 
    F=e**(-(((x - mu)/std)**2.)/2)/(2.*pi)**(0.5)/std
    return F
    
def Fit(h,x,funkcija): #fitanje na vrednosti histogramov
    p=[0,1] #začetni približek [mu,std]
    coeff, var_matrix = curve_fit(funkcija, x, h, p0=p)
    return coeff #[mu,std]
      
def fdata(x,mu,std): #funkcija za računanje vrednosti gaussove porazdelitve v sredinah položajev binov
    e=2.718281828459045    
    pi=3.141592653589793 
    F=e**(-(((x - mu)/std)**2.)/2)/(2.*pi)**(0.5)/std
    return F
#____________________________________________________ 
# generiranje števil in statistična analiza

#import cProfile
#cProfile.run("np.random.randn(10000000)")
#cProfile.run("np.random.rand(10000000)")

M=5 #št. ponovitev za porazdelitev statistike

SHfi=np.empty(M) 
SHth=np.empty(M) 
SHCth=np.empty(M) 
SHC2th=np.empty(M) 

SH01=np.empty(M) 
SH11=np.empty(M) 
SH02=np.empty(M) 
SH12=np.empty(M) 
SH22=np.empty(M) 
SH03=np.empty(M) 

n=np.empty(M) 

i=0
while i<M:

    N=100*10**i  # velikost vzorcev
    n[i]=N
       
    a1=np.random.rand(N)  # generiranje števil uniform
    a2=np.random.rand(N)
    
    fi=2.*pi*a1
    theta=np.arccos(2.*a2-1.)
    
    Hfi, xfi, binEdge_fi, Lfi=Hdata(fi)  
    Hth, xth, binEdge_th, Lth=Hdata(theta)
    HCth, xCth, binEdge_Cth, LCth=Hdata(np.cos(theta))
    HC2th, xC2th, binEdge_C2th, LC2th =Hdata(np.cos(theta)**2.)
    
    Hh01, xh01, binEdge_xh01, Lh01 =Hdata(np.real(sc.special.sph_harm(0, 1, fi, theta)))
    Hh11, xh11, binEdge_xh11, Lh11 =Hdata(np.real(sc.special.sph_harm(1, 1, fi, theta)))
    Hh02, xh02, binEdge_xh02, Lh02 =Hdata(np.real(sc.special.sph_harm(0, 2, fi, theta)))
    Hh12, xh12, binEdge_xh12, Lh12 =Hdata(np.real(sc.special.sph_harm(1, 2, fi, theta)))
    Hh22, xh22, binEdge_xh22, Lh22 =Hdata(np.real(sc.special.sph_harm(2, 2, fi, theta)))
    Hh03, xh03, binEdge_xh03, Lh03 =Hdata(np.real(sc.special.sph_harm(0, 3, fi, theta)))
    Hh04, xh04, binEdge_xh04, Lh00 =Hdata(np.real(sc.special.sph_harm(0, 4, fi, theta)))
    
    
    SHfi[i]=np.sum(Hfi* np.diff(binEdge_fi)* xfi )
    SHth[i]=np.sum(Hth* np.diff(binEdge_th)* xth )
    SHCth[i]=np.sum(HCth* np.diff(binEdge_Cth)* xCth )
    SHC2th[i]=np.sum(HC2th* np.diff(binEdge_C2th)* xC2th )
    
    SH01[i]=np.sum(Hh01* np.diff(binEdge_xh01)* xh01 )
    SH11[i]=np.sum(Hh11* np.diff(binEdge_xh11)* xh11 )
    SH02[i]=np.sum(Hh02* np.diff(binEdge_xh02)* xh02 )
    SH12[i]=np.sum(Hh12* np.diff(binEdge_xh12)* xh12 )
    SH22[i]=np.sum(Hh22* np.diff(binEdge_xh22)* xh22 )
    SH03[i]=np.sum(Hh03* np.diff(binEdge_xh03)* xh03 )
    
    i=i+1 


###### ____izris_grafov_in_histogramov______________________________
crta=['k','c','m','y','b','r']

F10=plt.figure(10)
F10=plt.subplot(2, 2, 1 ) 
plt.step(xfi,Hfi,'m',alpha = 0.5,label=r'$\langle\phi\rangle=$'+'{:.{}f}'.format(SHfi[M-1], 3 ))
plt.xlabel(r'$ \phi $')
plt.ylabel('n')
plt.legend(loc='best')
F10=plt.subplot(2, 2, 2 )  
plt.step(xth,Hth,'c',alpha = 0.5,label=r'$\langle\vartheta\rangle=$'+'{:.{}f}'.format(SHth[M-1], 3 ) )
plt.xlabel(r'$ \vartheta $')
plt.ylabel('n')
plt.legend(loc='best')
F10=plt.subplot(2, 2, 3 )  
plt.step(xCth,HCth,'c',alpha = 0.5,label=r'$\langle\cos(\vartheta)\rangle=$'+'{:.{}f}'.format(SHCth[M-1], 4 ) )
plt.xlabel(r'$ \cos(\vartheta) $')
plt.ylabel('n')
plt.legend(loc='best')
F10=plt.subplot(2, 2, 4 )  
plt.step(xC2th,HC2th,'c',alpha = 0.5,label=r'$\langle\cos^2(\vartheta)\rangle=$'+'{:.{}f}'.format(SHC2th[M-1], 4 ) )
plt.xlabel(r'$ \cos^2(\vartheta) $')
plt.ylabel('n')
plt.legend(loc='best')
plt.suptitle('Normirani histogrami porazdelitev greneriranih naključnih smeri in nekaterih momentov '+'(N='+str(N)+')',fontsize=16)

F20=plt.figure(20)
F20=plt.subplot(3, 2, 1 )  
plt.step(xh01,Hh01,'c',alpha = 0.5,label=r'$\langle Y^0_1 \rangle=$'+'{:.{}f}'.format(SH01[M-1], 5 ) )
plt.xlabel(r'$ Y^0_1  $')
plt.ylabel('n')
plt.legend(loc='best')
F20=plt.subplot(3, 2, 2 )  
plt.step(xh11,Hh11,'c',alpha = 0.5,label=r'$\langle Y^1_1 \rangle=$'+'{:.{}f}'.format(SH11[M-1], 5 ))
plt.xlabel(r'$  Y^1_1 $')
plt.ylabel('n')
plt.legend(loc='best')
F20=plt.subplot(3, 2, 3 )  
plt.step(xh02,Hh02,'c',alpha = 0.5,label=r'$\langle Y^0_2 \rangle=$'+'{:.{}f}'.format(SH02[M-1], 5 ))
plt.xlabel(r'$  Y^0_2 $')
plt.ylabel('n')
plt.legend(loc='best')
F20=plt.subplot(3, 2, 4 )  
plt.step(xh12,Hh12,'c',alpha = 0.5,label=r'$\langle Y^1_2 \rangle=$'+'{:.{}f}'.format(SH12[M-1], 5 ))
plt.xlabel(r'$  Y^1_2  $')
plt.ylabel('n')
plt.legend(loc='best')
F20=plt.subplot(3, 2, 5 )  
plt.step(xh22,Hh22,'c',alpha = 0.5,label=r'$\langle Y^2_2 \rangle=$'+'{:.{}f}'.format(SH22[M-1], 5 ))
plt.xlabel(r'$ Y^2_2 $')
plt.ylabel('n')
plt.legend(loc='best')
F20=plt.subplot(3, 2, 6 )  
plt.step(xh03,Hh03,'c',alpha = 0.5,label=r'$\langle Y^0_3 \rangle=$'+'{:.{}f}'.format(SH03[M-1], 5 ))
plt.xlabel(r'$ Y^0_3 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.suptitle('Normirani histogrami porazdelitev momentov greneriranih naključnih smeri '+'(N='+str(N)+')',fontsize=16)


F30=plt.figure(30)
F30=plt.subplot(2, 1, 1 ) 
plt.plot(n,abs(SHfi-pi),'c',alpha = 0.5,label=r'$\langle\phi\rangle - \pi$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(SHth-(pi/2.)),'m',alpha = 0.5,label=r'$\langle\vartheta\rangle-\pi/2$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(SHCth),'y',alpha = 0.5,label=r'$\langle\cos(\vartheta)\rangle$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(SHC2th-(1./3.)),'k',alpha = 0.5,label=r'$\langle\cos^2(\vartheta)\rangle-1/3$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(' N ')
plt.title('Absolutne vrednosti momentov generiranih koordinat za različno velike vzorce')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=1)
F30=plt.subplot(2, 1, 2 )  
plt.plot(n,abs(SH01),'y',alpha = 0.5,label=r'$\langle  Y^0_1 \rangle$')#+'{:.{}f}'.format(SHth, 3 ) )
plt.plot(n,abs(SH11),'c',alpha = 0.5,label=r'$\langle  Y^1_1 \rangle$')#+'{:.{}f}'.format(SHth, 3 ) )
plt.plot(n,abs(SH02),'k',alpha = 0.5,label=r'$\langle  Y^0_2 \rangle$')#+'{:.{}f}'.format(SHth, 3 ) )
plt.plot(n,abs(SH12),'g',alpha = 0.5,label=r'$\langle  Y^1_2 \rangle$')#+'{:.{}f}'.format(SHth, 3 ) )
plt.plot(n,abs(SH22),'r',alpha = 0.5,label=r'$\langle  Y^2_2 \rangle$')#+'{:.{}f}'.format(SHth, 3 ) )
plt.plot(n,abs(SH03),'b',alpha = 0.5,label=r'$\langle  Y^0_3 \rangle$')#+'{:.{}f}'.format(SHth, 3 ) )
plt.xscale('log')
plt.yscale('log')
plt.xlabel(' N ')
plt.title('Absolutne vrednosti momentov sferičnih harmonikov za različno velike vzorce')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=1)

#
#F2=plt.figure(1)
##if N>10000:
##    Ax1 = F2.add_subplot(2,2,1)
##else:
#F2=plt.subplot(2, 2, 1 )
#plt.hist(a1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{a1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.hist(a2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{a2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#if N==10000: 
#    plt.hist(loto, bins='auto',normed=True,facecolor="None",edgecolor='g',label=r'$\tt{loto}$ (N='+str(sum(LOTO))+r',$\nu$='+str(Lloto-1)+r',$\chi^2$='+'{:.{}f}'.format(X2HLOTO, 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.title('Normirani histogrami enakomerne porazdelitve '+'(N='+str(N)+',M='+str(M)+')')
#plt.xlabel('x')
#plt.ylabel('n')
##plt.ylabel('n')
#plt.legend(loc=4)
##if N>10000:
##    Ax2 = F2.add_subplot(2,2,1, sharex=Ax1, frameon=False)
##    plt.step(XLOTO/len(XLOTO),LOTO/np.mean(LOTO),color='g',label=r'$\tt{LOTO}$ '+r'($\chi^2$='+'{:.{}f}'.format(X2loto, 3 )+')',alpha = 0.5,where='pre')#,ls='step')  # arguments are passed to np.histogram
##    plt.step(CENKE/len(XLOTO),ENKE,color='k',label=r'$\tt{uniform}$ '+r'($\chi^2$='+'{:.{}f}'.format(X2enke, 3 )+')',alpha = 0.5,where='pre')#,ls='step')  # arguments are passed to np.histogram    
##    Ax2.yaxis.tick_right()
##    Ax2.yaxis.set_label_position("right")
##    Ax2.set_ylabel(r'$\tt{LOTO}$', color='g')
##    Ax2.set_ylim([0,1.2])
##    Ax2.tick_params('y', colors='g')
##    plt.legend(loc=4)
#
##if N=10000:
##    aAx1 = F2.add_subplot(2,2,2)
##else:
#F2=plt.subplot(2, 2, 2 )  
#plt.hist(a1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r'$\tt{a1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.hist(a2, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',label=r'$\tt{a2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#if N==10000:
#    plt.hist(loto, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='g',label=r'$\tt{loto}$ (N='+str(sum(LOTO))+r',D='+'{:.{}f}'.format(KuDloto, 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.title('Kumilativni normirani histogrami enakomerne porazdelitve  '+'(N='+str(N)+',M='+str(M)+')')
#plt.ylabel('kumulativen(n)')
#plt.xlabel('x')
#plt.legend(loc=4)
##if N==10000: 
##    aAx2 = F2.add_subplot(2,2,2, sharex=aAx1, frameon=False)
##    plt.step(XLOTO/len(XLOTO),CLOTO/np.sum(LOTO),color='g',label=r'$\tt{LOTO}$ '+'(D='+'{:.{}f}'.format(KuDloto, 3 )+')',alpha = 0.5,where='pre')#,ls='step')  # arguments are passed to np.histogram
##    aAx2.yaxis.tick_right()
##    aAx2.yaxis.set_label_position("right")
##    aAx2.set_ylabel(r'$\tt{LOTO}$', color='g')
##    aAx2.set_ylim([0,1.2])
##    aAx2.tick_params('y', colors='g')
##    plt.legend(loc=4)
#
#F2=plt.subplot(2, 2, 3 )
#plt.hist(z1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{z1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.hist(z2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{z2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.hist(K1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{K1}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.hist(a, bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{norm}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.title('Normirani histogrami normalne porazdelitve  '+'(N='+str(N)+',M='+str(M)+')')
#plt.xlabel('x')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.show()
#F2=plt.subplot(2, 2, 4 )
#plt.hist(z1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r'$\tt{z1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.hist(z2, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',label=r'$\tt{z2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
#plt.hist(K1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='r',label=r'$\tt{K1}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.hist(a, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='k',label=r'$\tt{norm}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.title('Kumilativni normirani histogrami normalne porazdelitve  '+'(N='+str(N)+',M='+str(M)+')')
#plt.ylabel('kumulativen(n)')
#plt.xlabel('x')
#plt.legend(loc='best')
#plt.show()
#
##F3=plt.figure(3)
##plt.hist(X2a,bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{chisquare(a) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
##plt.hist(X2a1,bins='auto',normed=True,facecolor="None",edgecolor='g',label=r'$\tt{chisquare(a1) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
##plt.hist(X2z1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{chisquare(z1) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
##plt.hist(X2z2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{chisquare(z2) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
##plt.hist(X2K1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{chisquare(K1) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
##plt.title('Histogrami porazdelitev vrednosti '+r'$\tt{\chi^2}$')
##plt.legend(loc=1)
##plt.show()
#
#F3=plt.figure(3)
#ax1 = F3.add_subplot(211)
##line1 = ax1.plot([1,3,4,5,2], 'o-')
#plt.hist(X2a,bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{chisquare(a,fit(a)) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
#plt.hist(X2z1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{chisquare(z1.fit(z1)) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.hist(X2z2, bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{chisquare(z2,fit(z2)) }$',alpha = 0.4,histtype='step') # arguments are passed to np.histogram
#plt.hist(X2K1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{chisquare(K1,fit(K1)) }$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
#plt.title('Normirani  histogrami porazdelitev vrednosti '+r'$\tt{\chi^2}$' +'(N='+str(N)+',M='+str(M)+')')
#ax1.set_xlabel(r'$\tt{\chi^2}$')
#ax1.set_xlim([0,1])
#ax1.set_ylabel('n')
##plt.ylabel("Left Y-Axis Data")
#plt.legend(loc=1)
## now, the second axes that shares the x-axis with the ax1
#ax2 = F3.add_subplot(211, sharex=ax1, frameon=False)
#plt.hist(X2a1,bins='auto',normed=True,facecolor="None",edgecolor='g',label=r"$\tt{chisquare(a1,'uniform') }$",alpha = 0.4,histtype='step') # arguments are passed to np.histogram
##line2 = ax2.plot([10,40,20,30,50], 'xr-')
#ax2.yaxis.tick_right()
#ax2.yaxis.set_label_position("right")
#ax2.set_xlim([0,1])
#ax2.set_ylabel(r'n iz $\tt{chisquare(a1) }$ podatkov', color='g')
#ax2.tick_params('y', colors='g')
#plt.legend(loc=4)
#plt.show()
#F3.add_subplot(212)
#plt.hist(La-1, bins='auto',normed=True,histtype='step',color='k',alpha = 0.5,label=r"$\tt{\nu_{a}}$")  # arguments are passed to np.histogram
#plt.hist(Lz1-1, bins='auto',normed=True,histtype='step',color='c',alpha = 0.5,label=r"$\tt{\nu_{z1}}$")  # arguments are passed to np.histogram
#plt.hist(Lz2-1, bins='auto',normed=True,histtype='step',color='m',alpha = 0.5,label=r"$\tt{\nu_{z2}}$")  # arguments are passed to np.histogram
#plt.hist(LK1-1, bins='auto',normed=True,histtype='step',color='r',alpha = 0.5,label=r"$\tt{\nu_{K1}}$")  # arguments are passed to np.histogram
#plt.hist(La1-1, bins='auto',normed=True,histtype='step',color='g',alpha = 0.5,label=r"$\tt{\nu_{a1}}$")  # arguments are passed to np.histogram
#plt.title(r'Prostostne stopnje $\nu$ histogramov iz katerih so računani '+r'$\tt{\chi^2}$ '+'(N='+str(N)+',M='+str(M)+')')
#plt.xlabel(r'$\nu$')
#plt.ylabel('n')
#plt.legend(loc=1)
#plt.show()
#
#
#F4=plt.figure(4)
#F4=plt.subplot(2, 1, 1 )
#plt.hist(KDz12, bins='auto',normed=True,histtype='step',color='y',alpha = 0.5,label=r"$\tt{ks\_2samp(z1,z2)}$")  # arguments are passed to np.histogram
#plt.hist(KDz1a, bins='auto',normed=True,histtype='step',color='c',alpha = 0.5,label=r"$\tt{ks\_2samp(z1,a)}$")  # arguments are passed to np.histogram
#plt.hist(KDz2a, bins='auto',normed=True,histtype='step',color='m',alpha = 0.5,label=r"$\tt{ks\_2samp(z2,a)}$")  # arguments are passed to np.histogram
#plt.hist(KDK1a, bins='auto',normed=True,histtype='step',color='r',alpha = 0.5,label=r"$\tt{ks\_2samp(K1,a)}$")  # arguments are passed to np.histogram
#plt.title('Normirani histogrami porazdelitev vrednosti D primerjalnega K-S testa  '+'(N='+str(N)+',M='+str(M)+')')
#plt.xlabel('D')
#plt.ylabel('n')
#plt.legend(loc=1)
#plt.show()
#F4=plt.subplot(2, 1, 2 )
#plt.hist(KnDa, bins='auto',normed=True,histtype='step',color='k',alpha = 0.4,label=r"$\tt{kstest(a, 'norm')}$")  # arguments are passed to np.histogram
#plt.hist(KnDz1, bins='auto',normed=True,histtype='step',color='c',alpha = 0.4,label=r"$\tt{kstest(z1, 'norm')}$")  # arguments are passed to np.histogram
#plt.hist(KnDz2, bins='auto',normed=True,histtype='step',color='m',alpha = 0.4,label=r"$\tt{kstest(z2, 'norm')}$")  # arguments are passed to np.histogram
#plt.hist(KnDK1, bins='auto',normed=True,histtype='step',color='r',alpha = 0.4,label=r"$\tt{kstest(K1, 'norm')}$")  # arguments are passed to np.histogram
#plt.hist(KuDa1, bins='auto',normed=True,histtype='step',color='b',alpha = 0.5,label=r"$\tt{kstest(a1, 'uniform')}$")  # arguments are passed to np.histogram
#plt.title('Normirani histogrami porazdelitev vrednosti D primerjalnega K-S testa  '+'(N='+str(N)+',M='+str(M)+')')
#plt.xlabel('D')
#plt.ylabel('n')
#plt.legend(loc=1)
#plt.show()
