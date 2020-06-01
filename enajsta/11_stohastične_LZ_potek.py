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

e=2.718281828459045    
pi=3.141592653589793 
#
#def BM(a1,a2): #Box-Muller
#    pi=3.141592653589793 
#    z1 = (-2 * np.log(a1))**(0.5) * np.sin(2 * pi * a2)
#    z2 = (-2 * np.log(a1))**(0.5) * np.cos(2 * pi * a2)
#    return z1,z2

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
     
#def Fdata(x,*p):  #funkcija za fitanje gaussove porazdelitve
#    mu,std=p
#    e=2.718281828459045    
#    pi=3.141592653589793 
#    F=e**(-(((x - mu)/std)**2.)/2)/(2.*pi)**(0.5)/std
#    return F
#    
#def Fit(h,x,funkcija): #fitanje na vrednosti histogramov
#    p=[0,1] #začetni približek [mu,std]
#    coeff, var_matrix = curve_fit(funkcija, x, h, p0=p)
#    return coeff #[mu,std]
#      
#def fdata(x,mu,std): #funkcija za računanje vrednosti gaussove porazdelitve v sredinah položajev binov
#    e=2.718281828459045    
#    pi=3.141592653589793 
#    F=e**(-(((x - mu)/std)**2.)/2)/(2.*pi)**(0.5)/std
#    return F
#____________________________________________________ 
# generiranje števil in statistična analiza

#import cProfile
#cProfile.run("np.random.randn(10000000)")
#cProfile.run("np.random.rand(10000000)")

#M=5 #št. ponovitev za porazdelitev statistike

Nt=np.empty(0)
Lt=np.empty(0)
Zt=np.empty(0)
T=np.empty(0)


potenca=2
N=10**potenca   # velikost vzorcev
  # generiranje števil uniform
#y=np.random.rand(N)
#z=np.random.rand(N)
#r=np.sqrt(x**2+y**2+z**2)            # radij

L0=50
Z0=200
t_smrti=0

dT=[0.001,0.01,0.1]
crta=['b','r','g','y','b','r']

n=np.empty(potenca) 
m=0

#Nt=np.append(Nt, N0)
T=np.append(T, t_smrti)

for j in range(0,3):
    
    L0=50.
    Z0=200.
    t_smrti=0    

    Lt=np.empty(0)
    Zt=np.empty(0)
    T=np.empty(0)

    Lt=np.append(Lt, L0)
    Zt=np.append(Zt, Z0)
    T=np.append(T, t_smrti)

#    N0=25

    dt=dT[j]
    
    i=0
    while  Zt[i]>0 and Lt[i]>0:
        
        dZ1=np.random.poisson(lam=5.0*Zt[i]*dt, size=None) #rojstvo
        dL1=np.random.poisson(lam=4.0*Lt[i]*dt, size=None)         #rojstvo
        dZ2=np.random.poisson(lam=4.0*Zt[i]*dt, size=None)  #smrt
        dL2=np.random.poisson(lam=5.0*Lt[i]*dt, size=None)        #smrt
        dZ3=np.random.poisson(lam=0.02*Zt[i]*Lt[i]*dt, size=None) #smrt
        dL3=np.random.poisson(lam=0.005*Zt[i]*Lt[i]*dt, size=None) #rojstvo
        
        Zt=np.append(Zt, Zt[i]+dZ1-dZ2-dZ3)
        Lt=np.append(Lt, Lt[i]+dL1-dL2+dL3)
        
#        dZ1=np.random.poisson(lam=5.0*Zt[i]*dt, size=None) #rojstvo
#        dL1=np.random.poisson(lam=4.0*Lt[i]*dt + 0.005*Zt[i]*Lt[i]*dt, size=None)         #rojstvo
#        dZ2=np.random.poisson(lam=4.0*Zt[i]*dt + 0.02*Zt[i]*Lt[i]*dt, size=None)  #smrt
#        dL2=np.random.poisson(lam=5.0*Lt[i]*dt, size=None)        #smrt
#        
#        Zt=np.append(Zt, Zt[i]+dZ1-dZ2)
#        Lt=np.append(Lt, Lt[i]+dL1-dL2)     
        
        
        t_smrti=t_smrti+dt
        T=np.append(T, t_smrti)
        
        
        i=i+1
        
    
    F50=plt.figure(50)
    plt.suptitle(r'Umiranje populacije računano z različnimi časovnimi koraki',fontsize=16)
    F50=plt.subplot(1, 2, 1 ) 
    plt.plot(T,Lt,color=crta[j],ls=':',alpha = 0.95,label=r'lisice $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
    plt.plot(T,Zt,color=crta[j],ls='--',alpha = 0.95,label=r'zajci $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
#    plt.yscale('log')
    plt.ylabel(' N(t) ')   
    plt.xlabel(' t ')
#    plt.xlim([0,6])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
    plt.legend(loc=0)
    
    F50=plt.subplot(1, 2, 2 )  
    plt.plot(Lt,Zt,color=crta[j],alpha = 0.95,label=r'$\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
#    plt.yscale('log')
    plt.ylabel(' Z(t) ')
    plt.xlabel(' L(t) ')
#    plt.xlim([0,6])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
    plt.legend(loc=0)

        
#    if i==10**(m+1)-1:     #integrali oziroma vsota vseh vrednosti normaliziranega histograma => povpreča 
#        Hkoc, xkoc, binEdge_koc, Lkoc=Hdata(np.append(rZ**2,rN**2))
#        Htelo, xtelo, binEdge_telo, Ltelo=Hdata(rN**2)  
#        HNEtelo, xNEtelo, binEdge_NEtelo, LNEtelo=Hdata(rZ**2)  
#        Hkrog, xkrog, binEdge_krog, Lkrog=Hdata(rkr**2)  
#        HvaljOS, x_valjOS, binEdge_valjOS, LvaljOS=Hdata(rvaljOS**2)  
#        HvaljPravok, x_valjPravok, binEdge_valjPravok, LvaljPravok=Hdata(rvaljPravok**2)  
#        
#        momentKoc=np.append(momentKoc,np.sum(Hkoc* np.diff(binEdge_koc)* xkoc ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#        momentTelo=np.append(momentTelo,np.sum(Htelo* np.diff(binEdge_telo)* xtelo ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
#        momentNETelo=np.append(momentNETelo,np.sum(HNEtelo* np.diff(binEdge_NEtelo)* xNEtelo ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
#        momentKrog=np.append(momentKrog,np.sum(Hkrog* np.diff(binEdge_krog)* xkrog ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#        momentvaljOS=np.append(momentvaljOS,np.sum(HvaljOS* np.diff(binEdge_valjOS)* x_valjOS ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#        momentvaljPravok=np.append(momentvaljPravok,np.sum(HvaljPravok* np.diff(binEdge_valjPravok)* x_valjPravok ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#    
#        volumenKoc=np.append(volumenKoc,len(rZ)+len(rN))
#        volumenTelo=np.append(volumenTelo,len(rN))
#        volumenNETelo=np.append(volumenNETelo,len(rZ))
#        volumenKrog=np.append(volumenKrog,len(rkr))
#        volumenValj=np.append(volumenValj,len(rvaljOS))
#    
#        print(m)
#        n[m]=j
#        m=m+1
        


###### ____izris_grafov_in_histogramov______________________________
#crta=['k','c','m','y','b','r']

#F10=plt.figure(10)
#plt.suptitle(r'Normirani histogrami porazdelitve momenta $r_z^2$ za N='+str(N)+' točk',fontsize=16)
#F10=plt.subplot(2, 3, 1 ) 
#plt.step(xkoc,Hkoc,'y',alpha = 0.5,label=r'$\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKoc[m-1], 5 ))
#plt.xlabel(r'$ r_z^2 $')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.title('Točke znotraj kocke prvega kvadranta')#+'(N='+str(N)+',M='+str(M)+')')
#F10=plt.subplot(2, 3, 2 ) 
#plt.step(xtelo,Htelo,'r',alpha = 0.5,label=r'$\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.xlabel(r'$ r_z^2 $')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.title('Točke znotraj preseka treh valjev')#+'(N='+str(N)+',M='+str(M)+')')
#F10=plt.subplot(2, 3, 3 ) 
#plt.step(xNEtelo,HNEtelo,'r',alpha = 0.5,label=r'$\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentNETelo[m-1], 5 ))
#plt.xlabel(r'$ r_z^2 $')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.title('Točke zunaj preseka treh valjev')#+'(N='+str(N)+',M='+str(M)+')')
#F10=plt.subplot(2, 3, 4 ) 
#plt.step(xkrog,Hkrog,'b',alpha = 0.5,label=r'$\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKrog[m-1], 5 ))
#plt.xlabel(r'$ r_z^2 $')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.title('Točke znotraj radija r<1')#+'(N='+str(N)+',M='+str(M)+')')
#F10=plt.subplot(2, 3, 5 ) 
#plt.step(x_valjOS,HvaljOS,'b',alpha = 0.5,label=r'$\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljOS[m-1], 5 ))
#plt.xlabel(r'$ r_z^2 $')
#plt.ylabel('n')
##plt.set_ylim(0,1)
#plt.legend(loc='best')
#plt.title('Točke pokoncnega valja')#+'(N='+str(N)+',M='+str(M)+')')
#F10=plt.subplot(2, 3, 6 ) 
#plt.step(x_valjPravok,HvaljPravok,'b',alpha = 0.5,label=r'$\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljPravok[m-1], 5 ))
#plt.xlabel(r'$ r_z^2 $')
#plt.ylabel('n')
#plt.legend(loc='best')
#plt.title('Točke ležečega valja')#+'(N='+str(N)+',M='+str(M)+')')

#F10=plt.subplot(2, 2, 2 )  
#plt.step(xth,Hth,'c',alpha = 0.5,label=r'$\langle\vartheta\rangle=$'+'{:.{}f}'.format(SHth[M-1], 3 ) )
#plt.xlabel(r'$ \vartheta $')
#plt.ylabel('n')
#plt.legend(loc='best')
#F10=plt.subplot(2, 2, 3 )  
#plt.step(xCth,HCth,'c',alpha = 0.5,label=r'$\langle\cos(\vartheta)\rangle=$'+'{:.{}f}'.format(SHCth[M-1], 4 ) )
#plt.xlabel(r'$ \cos(\vartheta) $')
#plt.ylabel('n')
#plt.legend(loc='best')
#F10=plt.subplot(2, 2, 4 )  
#plt.step(xC2th,HC2th,'c',alpha = 0.5,label=r'$\langle\cos^2(\vartheta)\rangle=$'+'{:.{}f}'.format(SHC2th[M-1], 4 ) )
#plt.xlabel(r'$ \cos^2(\vartheta) $')
#plt.ylabel('n')
#plt.legend(loc='best')

#F20=plt.figure(20)
#plt.suptitle('Naključno generirane točke v prostoru s kriteriji omejene na različne like')
#F20=plt.subplot(1, 3, 1 ) 

####
#F20=plt.figure(20)
#Axes3D = plt.axes(projection='3d')
#Axes3D.scatter(xN, yN, zN, zdir='z', s=1, c='r',alpha = 0.1, depthshade=False)
#plt.title(r'Točke prvega kvadranta znotraj preseka treh valjev $N_{\otimes}=$'+str(len(rN)))#+'(N='+str(N)+',M='+str(M)+')')
##F30=plt.subplot(1, 3, 2 ) 
#F30=plt.figure(30)
#Axes3D = plt.axes(projection='3d')
#Axes3D.scatter(xZ, yZ, zZ, zdir='z', s=1, c='c', alpha = 0.1, depthshade=False)
#plt.title(r'Točke prvega kvadranta zunaj preseka treh valjev $N_{!\otimes}=$'+str(len(rZ)))#+'(N='+str(N)+',M='+str(M)+')')
##F40=plt.subplot(1, 3, 3 ) 
#F40=plt.figure(40)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xZ, yZ, zZ, zdir='z', s=1, c='c', alpha = 0.1, depthshade=False)
#Axes3D.scatter(xN, yN, zN, zdir='z', s=1, c='r',alpha = 0.1, depthshade=False)
##plt.title('Točke znotraj radija r<1')#+'(N='+str(N)+',M='+str(M)+')')
#plt.title(r'Točke prvega kvadranta $N_{\boxdot}=$'+str(len(rZ)+len(rN)))#+'(N='+str(N)+',M='+str(M)+')')
#
#F41=plt.figure(41)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xvaljOS, yvaljOS, zvaljOS, zdir='z', s=1, c='g', alpha = 0.1, depthshade=False)
#plt.title(r'Točke prvega kvadranta pokončnega valja $N_{valj}=$'+str(len(rvaljOS)))#+'(N='+str(N)+',M='+str(M)+')')
#
#F42=plt.figure(42)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xvaljPravok, yvaljPravok, zvaljPravok, zdir='z', s=1, c='k', alpha = 0.1, depthshade=False)
#plt.title(r'Točke prvega kvadranta ležečega valja $N_{valj}=$'+str(len(rvaljPravok)))#+'(N='+str(N)+',M='+str(M)+')')
#
#
#F42=plt.figure(42)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=False)
#plt.title(r'Točke prvega kvadranta krogle $N_{krog}=$'+str(len(rkr)))#+'(N='+str(N)+',M='+str(M)+')')


#
##
#
#F50=plt.figure(50)
#F50=plt.subplot(2, 1, 1 ) 
#plt.plot(n,abs(momentKoc),'yo',alpha = 0.95,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentTelo),'ro',alpha = 0.95,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentNETelo),'co',alpha = 0.95,label=r'$zunaj\ preseka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentKrog),'bo',alpha = 0.95,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentvaljOS),'go',alpha = 0.95,label=r'$pokončni\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentvaljPravok),'ko',alpha = 0.95,label=r'$ležeči\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
#
#plt.xscale('log')
##plt.yscale('log')
#plt.xlabel(' N ')
#plt.title('Absolutne vrednosti momentov generiranih koordinat za različno velike vzorce')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=0)
#F50=plt.subplot(2, 1, 2 )  
##plt.plot(abs(np.diff(momentKoc)),'y',alpha = 0.5,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(abs(np.diff(momentTelo)),'r',alpha = 0.5,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(abs(np.diff(momentKrog)),'b',alpha = 0.5,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentKoc-0.666666666),'yo',alpha = 0.95,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentTelo-momentTelo[potenca-1]),'ro',alpha = 0.95,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentNETelo-momentNETelo[potenca-1]),'co',alpha = 0.95,label=r'$zunaj\ preseka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentKrog-0.4),'bo',alpha = 0.95,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentvaljOS-0.5),'go',alpha = 0.95,label=r'$pokončni\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(momentvaljPravok-0.5833333333),'ko',alpha = 0.95,label=r'$ležeči\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(' N ')
#plt.title('Absolutno odstopanje izračuna momenta od točne vrednosti za vsako povečanje velikosti vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=0)



#F60=plt.figure(60)
#F60=plt.subplot(2, 1, 1 ) 
#plt.plot(n,abs(volumenKoc)/abs(volumenKoc),'yo',alpha = 0.95,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(volumenKoc)/abs(volumenTelo),'ro',alpha = 0.95,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(volumenKoc)/abs(volumenNETelo),'co',alpha = 0.95,label=r'$zunaj\ preseka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(volumenKoc)/abs(volumenKrog),'bo',alpha = 0.95,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs(volumenKoc)/abs(volumenValj),'go',alpha = 0.95,label=r'$valj$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xscale('log')
##plt.yscale('log')
#plt.xlabel(' N ')
#plt.title('Absolutne vrednosti rezmerij prostornin teles napram kocki za različno velike vzorce')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=0)
#
#F60=plt.subplot(2, 1, 2 )  
##plt.plot(abs(np.diff(momentKoc)),'y',alpha = 0.5,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(abs(np.diff(momentTelo)),'r',alpha = 0.5,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(abs(np.diff(momentKrog)),'b',alpha = 0.5,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs((abs(volumenKoc)/abs(volumenTelo))-(abs(volumenKoc[potenca-1])/abs(volumenTelo[potenca-1]))),'ro',alpha = 0.95,label=r'$\frac{N_{\boxdot}}{N_{\otimes}} -\frac{V_{\boxdot}}{V_{\otimes}}$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs((abs(volumenKoc)/abs(volumenNETelo))-(abs(volumenKoc[potenca-1])/abs(volumenNETelo[potenca-1]))),'co',alpha = 0.95,label=r'$\frac{N_{\boxdot}}{!N_{\otimes}} -\frac{V_{\boxdot}}{!V_{\otimes}}$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs((abs(volumenKoc)/abs(volumenKrog))-1.9098593171),'bo',alpha = 0.95,label=r'$\frac{N_{\boxdot}}{N_{krogla}} -\frac{V_{\boxdot}}{V_{krogla}}$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(n,abs((abs(volumenKoc)/abs(volumenValj))-1.27323954474),'go',alpha = 0.95,label=r'$\frac{N_{\boxdot}}{N_{valj}} -\frac{V_{\boxdot}}{V_{valj}}$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(' N ')
#plt.title('Absolutno odstopanje izračuna volumna od točne vrednosti za vsako povečanje velikosti vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=0)
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
