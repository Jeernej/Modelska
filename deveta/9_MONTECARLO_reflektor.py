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

#xN=np.empty(0)
#yN=np.empty(0)
#zN=np.empty(0)
#xZ=np.empty(0)
#yZ=np.empty(0)
#zZ=np.empty(0)
#xkr=np.empty(0)
#ykr=np.empty(0)
#zkr=np.empty(0)
#xvaljOS=np.empty(0)
#yvaljOS=np.empty(0)
#zvaljOS=np.empty(0)
#xvaljPravok=np.empty(0)
#yvaljPravok=np.empty(0)
#zvaljPravok=np.empty(0)
#
#rkr=np.empty(0)
#rN=np.empty(0)
#rZ=np.empty(0)
#rvaljOS=np.empty(0)
#rvaljPravok=np.empty(0)
#
#momentKoc=np.empty(0) 
#momentTelo=np.empty(0) 
#momentNETelo=np.empty(0) 
#momentKrog=np.empty(0) 
#momentvaljOS=np.empty(0) 
#momentvaljPravok=np.empty(0) 
#volumenKoc=np.empty(0)
#volumenTelo=np.empty(0)
#volumenNETelo=np.empty(0)
#volumenKrog=np.empty(0)
#volumenValj=np.empty(0)

potenca=4
N=10**potenca   # velikost vzorcev

#x=np.random.rand(N)  # generiranje števil uniform
#y=np.random.rand(N)
#z=np.random.rand(N)
#r=np.sqrt(x**2+y**2+z**2)/np.sqrt(3)     # radij
#r=x**(1/3)

theta=np.random.rand(N)*pi*2
#theta=np.arccos(2*np.random.rand(N)-1)

#l=-r*np.cos(theta) + np.sqrt( 1 - (r*np.sin(theta))**2) # prepotovana pot

#mu0=1
#L=-np.log(1-y)/mu0 # prepotovana pot


n=np.empty(potenca) 

#MU=np.empty(50) 
#for k in range(0,50): MU[k]=(k+0.001)/5

odbiti6=0
odbiti5=0
odbiti4=0
odbiti3=0
odbiti2=0
odbiti1=0
pobegli1=0    
pobegli2=0    
pobegli3=0    
pobegli4=0    
pobegli5=0    
pobegli6=0    

Odbiti6=np.empty(0)
Odbiti5=np.empty(0)
Odbiti4=np.empty(0)
Odbiti3=np.empty(0)
Odbiti2=np.empty(0)
Odbiti1=np.empty(0)
Pobegli1=np.empty(0)
Pobegli2=np.empty(0)
Pobegli3=np.empty(0)
Pobegli4=np.empty(0)
Pobegli5=np.empty(0)
Pobegli6=np.empty(0)

KOT_odbiti6=np.empty(0) 
KOT_odbiti5=np.empty(0) 
KOT_odbiti4=np.empty(0) 
KOT_odbiti3=np.empty(0) 
KOT_odbiti2=np.empty(0) 
KOT_odbiti1=np.empty(0) 
KOT_pobegli1=np.empty(0) 
KOT_pobegli2=np.empty(0) 
KOT_pobegli3=np.empty(0) 
KOT_pobegli4=np.empty(0) 
KOT_pobegli5=np.empty(0) 
KOT_pobegli6=np.empty(0) 

sipanje_odbiti6=np.empty(0) 
sipanje_odbiti5=np.empty(0) 
sipanje_odbiti4=np.empty(0) 
sipanje_odbiti3=np.empty(0) 
sipanje_odbiti2=np.empty(0) 
sipanje_odbiti1=np.empty(0) 
sipanje_pobegli1=np.empty(0) 
sipanje_pobegli2=np.empty(0) 
sipanje_pobegli3=np.empty(0) 
sipanje_pobegli4=np.empty(0) 
sipanje_pobegli5=np.empty(0) 
sipanje_pobegli6=np.empty(0) 



m=0
for i in range(0,N):
    
    sipanje=0
    psi=0
    L=-np.log(1-np.random.rand())
    
    L0=0
    L1=0
    L2=0
    L3=0
    L4=0
    L5=0
    L6=0
    
    while L<=10:

        if L>=0.1 and L1==0:  # za debeilno 0.1 sipalne dolžine
            
            pobegli1=pobegli1+1 #število prehodov 
#            KOT_pobegli1=np.append(KOT_pobegli1,psi) #kot izstopa prehoda
            sipanje_pobegli1=np.append(sipanje_pobegli1,sipanje)  #število sipanj do prehoda
            L1=1
    
        if L>=0.5 and L2==0:  # za debeilno 0.1 sipalne dolžine
            
            pobegli2=pobegli2+1 #število prehodov 
#            KOT_pobegli2=np.append(KOT_pobegli2,psi) #kot izstopa prehoda
            sipanje_pobegli2=np.append(sipanje_pobegli2,sipanje)  #število sipanj do prehoda
            L2=1

        if L>=1 and L3==0:  # za debeilno 0.1 sipalne dolžine
            
            pobegli3=pobegli3+1 #število prehodov 
#            KOT_pobegli3=np.append(KOT_pobegli3,psi) #kot izstopa prehoda
            sipanje_pobegli3=np.append(sipanje_pobegli3,sipanje)  #število sipanj do prehoda
            L3=1
            
        if L>=2 and L4==0:  # za debeilno 0.1 sipalne dolžine
            
            pobegli4=pobegli4+1 #število prehodov 
#            KOT_pobegli4=np.append(KOT_pobegli4,psi) #kot izstopa prehoda
            sipanje_pobegli4=np.append(sipanje_pobegli4,sipanje)  #število sipanj do prehoda
            L4=1


        if L>=5 and L5==0:  # za debeilno 0.1 sipalne dolžine
            
            pobegli5=pobegli5+1 #število prehodov 
#            KOT_pobegli5=np.append(KOT_pobegli5,psi) #kot izstopa prehoda
            sipanje_pobegli5=np.append(sipanje_pobegli5,sipanje)  #število sipanj do prehoda
            L5=1
            
            
        sipanje=sipanje+1
        
        if sipanje%2 == 0: L = L - np.log(1-np.random.rand())
        if sipanje%2 != 0: L = L + np.log(1-np.random.rand())
            
#        L = L - np.log(1-np.random.rand())
        
#        psi=psi+np.random.rand()*2*pi   
#        L = L - np.log(1-np.random.rand()) * np.cos(psi)
#        print('sipanje')

        if L<=0 :
            
            if L1==0:
            
                odbiti1=odbiti1+1
        #            KOT_odbiti1=np.append(KOT_odbiti1,psi)
                sipanje_odbiti1=np.append(sipanje_odbiti1,sipanje)
                L1=1
#                print('odbit')
            if L2==0:
            
                odbiti2=odbiti2+1
        #            KOT_odbiti2=np.append(KOT_odbiti2,psi)
                sipanje_odbiti2=np.append(sipanje_odbiti2,sipanje)
                L2=1
#                print('odbit')
            if L3==0:
            
                odbiti3=odbiti3+1
        #            KOT_odbiti3=np.append(KOT_odbiti3,psi)
                sipanje_odbiti3=np.append(sipanje_odbiti3,sipanje)
                L3=1
#                print('odbit')
            if L4==0:
            
                odbiti4=odbiti4+1
        #            KOT_odbiti4=np.append(KOT_odbiti4,psi)
                sipanje_odbiti4=np.append(sipanje_odbiti4,sipanje)
                L4=1
            if L5==0:
            
                odbiti5=odbiti5+1
        #            KOT_odbiti5=np.append(KOT_odbiti5,psi)
                sipanje_odbiti5=np.append(sipanje_odbiti5,sipanje)
                L5=1#                print('odbit')
            if L6==0:
            
                odbiti6=odbiti6+1
        #            KOT_odbiti6=np.append(KOT_odbiti6,psi)
                sipanje_odbiti6=np.append(sipanje_odbiti6,sipanje)
                L6=1
            break
        
        
        
    if L>=10 and L6==0:  # za debeilno 0.1 sipalne dolžine
        
        pobegli6=pobegli6+1 #število prehodov 
#        KOT_pobegli6=np.append(KOT_pobegli6,psi) #kot izstopa prehoda
        sipanje_pobegli6=np.append(sipanje_pobegli6,sipanje)  #število sipanj do prehoda
        L6=1              

   
    if i==10**(m+1)-1:     #integrali oziroma vsota vseh vrednosti normaliziranega histograma => povpreča 

       
        Odbiti1=np.append(Odbiti1,odbiti1/i)
        Odbiti2=np.append(Odbiti2,odbiti2/i)
        Odbiti3=np.append(Odbiti3,odbiti3/i)
        Odbiti4=np.append(Odbiti4,odbiti4/i)
        Odbiti5=np.append(Odbiti5,odbiti5/i)
        Odbiti6=np.append(Odbiti6,odbiti6/i)
        Pobegli1=np.append(Pobegli1,pobegli1/i)
        Pobegli2=np.append(Pobegli2,pobegli2/i)
        Pobegli3=np.append(Pobegli3,pobegli3/i)
        Pobegli4=np.append(Pobegli4,pobegli4/i)
        Pobegli5=np.append(Pobegli5,pobegli5/i)
        Pobegli6=np.append(Pobegli6,pobegli6/i)
        
        print(m)
        n[m]=i
        m=m+1
        
#        print(pobegli/N)

Hsipanje_odbiti1, xsipanje_odbiti1, binEdge_sipanje_odbiti1, Lsipanje_odbiti1=Hdata(sipanje_odbiti1)
Hsipanje_odbiti2, xsipanje_odbiti2, binEdge_sipanje_odbiti2, Lsipanje_odbiti2=Hdata(sipanje_odbiti2)
Hsipanje_odbiti3, xsipanje_odbiti3, binEdge_sipanje_odbiti3, Lsipanje_odbiti3=Hdata(sipanje_odbiti3)
Hsipanje_odbiti4, xsipanje_odbiti4, binEdge_sipanje_odbiti4, Lsipanje_odbiti4=Hdata(sipanje_odbiti4)
Hsipanje_odbiti5, xsipanje_odbiti5, binEdge_sipanje_odbiti5, Lsipanje_odbiti5=Hdata(sipanje_odbiti5)
Hsipanje_odbiti6, xsipanje_odbiti6, binEdge_sipanje_odbiti6, Lsipanje_odbiti6=Hdata(sipanje_odbiti6)
Hsipanje_pobegli1, xsipanje_pobegli1, binEdge_sipanje_pobegli1, Lsipanje_pobegli1=Hdata(sipanje_pobegli1)  
Hsipanje_pobegli2, xsipanje_pobegli2, binEdge_sipanje_pobegli2, Lsipanje_pobegli2=Hdata(sipanje_pobegli2)  
Hsipanje_pobegli3, xsipanje_pobegli3, binEdge_sipanje_pobegli3, Lsipanje_pobegli3=Hdata(sipanje_pobegli3)  
Hsipanje_pobegli4, xsipanje_pobegli4, binEdge_sipanje_pobegli4, Lsipanje_pobegli4=Hdata(sipanje_pobegli4)  
Hsipanje_pobegli5, xsipanje_pobegli5, binEdge_sipanje_pobegli5, Lsipanje_pobegli5=Hdata(sipanje_pobegli5)  
Hsipanje_pobegli6, xsipanje_pobegli6, binEdge_sipanje_pobegli6, Lsipanje_pobegli6=Hdata(sipanje_pobegli6)  
#
#HKOT_odbiti, xKOT_odbiti, binEdge_KOT_odbiti, LKOT_odbiti=Hdata(KOT_odbiti)  
#HKOT_pobegli1 , xKOT_pobegli1 , binEdge_KOT_pobegli1, LKOT_pobegli1=Hdata(KOT_pobegli1)  
#HKOT_pobegli2 , xKOT_pobegli2 , binEdge_KOT_pobegli2, LKOT_pobegli2=Hdata(KOT_pobegli2)  
#HKOT_pobegli3 , xKOT_pobegli3 , binEdge_KOT_pobegli3, LKOT_pobegli3=Hdata(KOT_pobegli3)  
#HKOT_pobegli4 , xKOT_pobegli4 , binEdge_KOT_pobegli4, LKOT_pobegli4=Hdata(KOT_pobegli4)  
#HKOT_pobegli5 , xKOT_pobegli5 , binEdge_KOT_pobegli5, LKOT_pobegli5=Hdata(KOT_pobegli5)  
#HKOT_pobegli6 , xKOT_pobegli6 , binEdge_KOT_pobegli6, LKOT_pobegli6=Hdata(KOT_pobegli6)  


###### ____izris_grafov_in_histogramov______________________________
#crta=['k','c','m','y','b','r']

#X=list(range(int(max(sipanje_pobegli6))))

F101=plt.figure(101)
#plt.step(xsipanje_odbiti,Hsipanje_odbiti,'k',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xsipanje_pobegli1,Hsipanje_pobegli1,'g',alpha = 0.5,label=r'pobegli $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xsipanje_pobegli2,Hsipanje_pobegli2,'r',alpha = 0.5,label=r'pobegli $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_pobegli3,Hsipanje_pobegli3,'b',alpha = 0.5,label=r'pobegli $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_pobegli4,Hsipanje_pobegli4,'m',alpha = 0.5,label=r'pobegli $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_pobegli5,Hsipanje_pobegli5,'k',alpha = 0.5,label=r'pobegli $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_pobegli6,Hsipanje_pobegli6,'y',alpha = 0.5,label=r'pobegli $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.xlabel(r'število sipanj')
#plt.ylabel(r'$dP/dl$')
plt.ylabel('n')
#plt.xlim(0,6)
#plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.title('Porazdelitev pobeglih nevtronov \n po številu sipanj do izstopa iz reflektorja; računano za N='+str(N))#+'(N='+str(N)+',M='+str(M)+')')

F102=plt.figure(102)
#plt.step(xtelo,Htelo,'k',alpha = 0.5,label=r'$\langle l \rangle=$'+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xtelo,Htelo,'r',alpha = 0.5,label=r'$l_{(r,\theta)}$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xkrog,Hkrog,'b',alpha = 0.5,label=r'$l_{(\mu=1)}$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xsipanje_odbiti1,Hsipanje_odbiti1,'g',alpha = 0.5,label=r'odbiti $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xsipanje_odbiti2,Hsipanje_odbiti2,'r',alpha = 0.5,label=r'odbiti $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_odbiti3,Hsipanje_odbiti3,'b',alpha = 0.5,label=r'odbiti $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_odbiti4,Hsipanje_odbiti4,'m',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_odbiti5,Hsipanje_odbiti5,'k',alpha = 0.5,label=r'odbiti $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.step(xsipanje_odbiti6,Hsipanje_odbiti6,'y',alpha = 0.5,label=r'odbiti $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.xlabel(r'število sipanj')
#plt.ylabel(r'$dP/dl$')
plt.ylabel('n')
#plt.xlim(0,6)
#plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.title('Porazdelitev odbitih nevtronov \n po številu sipanj do izstopa iz reflektorja; računano za N='+str(N))#+'(N='+str(N)+',M='+str(M)+')')

#F201=plt.figure(201)
##plt.step(xtelo,Htelo,'k',alpha = 0.5,label=r'$\langle l \rangle=$'+'{:.{}f}'.format(momentTelo[m-1], 5 ))
##plt.step(xtelo,Htelo,'r',alpha = 0.5,label=r'$l_{(r,\theta)}$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
##plt.step(xkrog,Hkrog,'b',alpha = 0.5,label=r'$l_{(\mu=1)}$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti,HKOT_odbiti,'k',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_pobegli1,HKOT_pobegli1,'g',alpha = 0.5,label=r'pobegli $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_pobegli2,HKOT_pobegli2,'r',alpha = 0.5,label=r'pobegli $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_pobegli2,HKOT_pobegli3,'b',alpha = 0.5,label=r'pobegli $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_pobegli4,HKOT_pobegli4,'m',alpha = 0.5,label=r'pobegli $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_pobegli5,HKOT_pobegli5,'c',alpha = 0.5,label=r'pobegli $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_pobegli6,HKOT_pobegli6,'y',alpha = 0.5,label=r'pobegli $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.xlabel(r'$\cos(\theta)$')
##plt.ylabel(r'$dP/dl$')
#plt.ylabel('n')
##plt.xlim(0,6)
#plt.legend(loc='best')
#plt.title(r'Porazdelitev po vrednostih izstopnih kotov pobegov iz reflektorja; računano za N='+str(N)+' nevtronov')#+'(N='+str(N)+',M='+str(M)+')')

#F202=plt.figure(202)
##plt.step(xtelo,Htelo,'k',alpha = 0.5,label=r'$\langle l \rangle=$'+'{:.{}f}'.format(momentTelo[m-1], 5 ))
##plt.step(xtelo,Htelo,'r',alpha = 0.5,label=r'$l_{(r,\theta)}$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
##plt.step(xkrog,Hkrog,'b',alpha = 0.5,label=r'$l_{(\mu=1)}$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
##plt.step(xKOT_odbiti,HKOT_odbiti,'k',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti1,HKOT_odbiti1,'g',alpha = 0.5,label=r'pobegli $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti2,HKOT_odbiti2,'r',alpha = 0.5,label=r'pobegli $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti3,HKOT_odbiti3,'b',alpha = 0.5,label=r'pobegli $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti4,HKOT_odbiti4,'m',alpha = 0.5,label=r'pobegli $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti5,HKOT_odbiti5,'c',alpha = 0.5,label=r'pobegli $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.step(xKOT_odbiti6,HKOT_odbiti6,'y',alpha = 0.5,label=r'pobegli $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.xlabel(r'$\cos(\theta)$')
##plt.ylabel(r'$dP/dl$')
#plt.ylabel('n')
##plt.xlim(0,6)
#plt.legend(loc='best')
#plt.title(r'Porazdelitev po vrednostih izstopnih kotov odbojev iz reflektorja; računano za N='+str(N)+' nevtronov')#+'(N='+str(N)+',M='+str(M)+')')



MU=np.array([0.1,0.5,1,2,5,10])
L_pobegli=np.array([pobegli1/N,pobegli2/N,pobegli3/N,pobegli4/N,pobegli5/N,pobegli6/N])
L_odbiti=np.array([odbiti1/N,odbiti2/N,odbiti3/N,odbiti4/N,odbiti5/N,odbiti6/N])

F30=plt.figure(30)
#plt.plot(L,odbiti/N,'k',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(MU,L_pobegli,'k',alpha = 0.5,label=r'pobegli')# $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(MU,L_odbiti,'k:',alpha = 0.5,label=r'odbiti ')#$\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(L,pobegli2/N,'r',alpha = 0.5,label=r'pobegli $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(L,pobegli3/N,'b',alpha = 0.5,label=r'pobegli $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(L,pobegli4/N,'m',alpha = 0.5,label=r'pobegli $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(L,pobegli5/N,'c',alpha = 0.5,label=r'pobegli $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(L,pobegli6/N,'y',alpha = 0.5,label=r'pobegli $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.xlabel(r'$\mu=d_0/\lambda$')
#plt.ylabel(r'$dP/dl$')
plt.ylabel(r'$P_{\mu}$')
#plt.xlim(0,6)
plt.legend(loc='best')
plt.title('Verjetnosti za prehod (odboj) nevtronov \n pri različnih debelinah reflektorja; računano za N='+str(N)+' nevtronov')#+'(N='+str(N)+',M='+str(M)+')')


F60=plt.figure(60)
F60=plt.subplot(2, 1, 1 )  
plt.plot(n,Odbiti4,'ko',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(n,Pobegli1,'go',alpha = 0.5,label=r'pobegli $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(n,Pobegli2,'ro',alpha = 0.5,label=r'pobegli $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli3,'bo',alpha = 0.5,label=r'pobegli $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli4,'mo',alpha = 0.5,label=r'pobegli $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli5,'co',alpha = 0.5,label=r'pobegli $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli6,'yo',alpha = 0.5,label=r'pobegli $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.xlabel(' N ')
plt.xscale('log')
#plt.ylabel(r'$dP/dl$')
plt.ylabel(r'$P_{\mu}$')
#plt.xlim(0,6)
plt.legend(loc=2)
plt.title(r'Verjetnosti za prehod (odboj) nevtronov za različne velikosti vzorca')#+'(N='+str(N)+',M='+str(M)+')')

F60=plt.subplot(2, 1, 2 )  
plt.plot(n,Odbiti4-Odbiti4[-1],'ko',alpha = 0.5,label=r'odbiti $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(n,Pobegli1-Pobegli1[-1],'go',alpha = 0.5,label=r'pobegli $\mu=0.1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
#plt.plot(n,Pobegli2-Pobegli2[-1],'ro',alpha = 0.5,label=r'pobegli $\mu=0.5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli3-Pobegli3[-1],'bo',alpha = 0.5,label=r'pobegli $\mu=1$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli4-Pobegli4[-1],'mo',alpha = 0.5,label=r'pobegli $\mu=2$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli5-Pobegli5[-1],'co',alpha = 0.5,label=r'pobegli $\mu=5$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.plot(n,Pobegli6-Pobegli6[-1],'yo',alpha = 0.5,label=r'pobegli $\mu=10$')#+'{:.{}f}'.format(momentTelo[m-1], 5 ))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(' N ')
plt.ylabel(r'$P_{\mu}-P_{\mu}(N=10^6)$')
plt.title('Absolutno odstopanje izračuna verjetnosti za prehod (odboj) od vrednosti dobljene pri velikosti vzorca N='+str(N))
plt.legend(loc=2)

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
#
#
#
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
