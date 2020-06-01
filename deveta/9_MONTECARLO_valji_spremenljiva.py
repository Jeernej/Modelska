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
r0=1

xN=np.empty(0)
yN=np.empty(0)
zN=np.empty(0)
xZ=np.empty(0)
yZ=np.empty(0)
zZ=np.empty(0)
xkr=np.empty(0)
ykr=np.empty(0)
zkr=np.empty(0)
xvaljOS=np.empty(0)
yvaljOS=np.empty(0)
zvaljOS=np.empty(0)
xvaljPravok=np.empty(0)
yvaljPravok=np.empty(0)
zvaljPravok=np.empty(0)

rm=np.empty(0)

rkr1=np.empty(0)
rkr2=np.empty(0)
rkr3=np.empty(0)
rkr4=np.empty(0)
rkr5=np.empty(0)
rkr6=np.empty(0)
rN1=np.empty(0)
rN2=np.empty(0)
rN3=np.empty(0)
rN4=np.empty(0)
rN5=np.empty(0)
rN6=np.empty(0)
rZ1=np.empty(0)
rZ2=np.empty(0)
rZ3=np.empty(0)
rZ4=np.empty(0)
rZ5=np.empty(0)
rZ6=np.empty(0)
rvaljOS1=np.empty(0)
rvaljOS2=np.empty(0)
rvaljOS3=np.empty(0)
rvaljOS4=np.empty(0)
rvaljOS5=np.empty(0)
rvaljOS6=np.empty(0)
rvaljPravok1=np.empty(0)
rvaljPravok2=np.empty(0)
rvaljPravok3=np.empty(0)
rvaljPravok4=np.empty(0)
rvaljPravok5=np.empty(0)
rvaljPravok6=np.empty(0)

momentKoc1=np.empty(0) 
momentKoc2=np.empty(0) 
momentKoc3=np.empty(0) 
momentKoc4=np.empty(0) 
momentKoc5=np.empty(0) 
momentKoc6=np.empty(0) 
momentTelo1=np.empty(0) 
momentTelo2=np.empty(0) 
momentTelo3=np.empty(0) 
momentTelo4=np.empty(0) 
momentTelo5=np.empty(0) 
momentTelo6=np.empty(0) 
momentNETelo1=np.empty(0) 
momentNETelo2=np.empty(0) 
momentNETelo3=np.empty(0) 
momentNETelo4=np.empty(0) 
momentNETelo5=np.empty(0) 
momentNETelo6=np.empty(0) 
momentKrog1=np.empty(0) 
momentKrog2=np.empty(0) 
momentKrog3=np.empty(0) 
momentKrog4=np.empty(0) 
momentKrog5=np.empty(0) 
momentKrog6=np.empty(0) 
momentvaljOS1=np.empty(0) 
momentvaljOS2=np.empty(0) 
momentvaljOS3=np.empty(0) 
momentvaljOS4=np.empty(0) 
momentvaljOS5=np.empty(0) 
momentvaljOS6=np.empty(0) 
momentvaljPravok1=np.empty(0) 
momentvaljPravok2=np.empty(0) 
momentvaljPravok3=np.empty(0) 
momentvaljPravok4=np.empty(0) 
momentvaljPravok5=np.empty(0) 
momentvaljPravok6=np.empty(0) 


potenca=6
N=10**potenca   # velikost vzorcev
x=np.random.rand(N)  # generiranje števil uniform
y=np.random.rand(N)
z=np.random.rand(N)
#r=np.sqrt(x**2+y**2+z**2)            # radij

n=np.empty(potenca) 
m=0
for i in range(0,N):

    mr1=((x[i]**2+y[i]**2+z[i]**2)/r0)**0.2      # obtežitev zaradi gostote glede na radij okrog izhodišča
    mr2=((x[i]**2+y[i]**2+z[i]**2)/r0)**0.5      # obtežitev zaradi gostote glede na radij okrog izhodišča
    mr3=((x[i]**2+y[i]**2+z[i]**2)/r0)**1      # obtežitev zaradi gostote glede na radij okrog izhodišča
    mr4=((x[i]**2+y[i]**2+z[i]**2)/r0)**2      # obtežitev zaradi gostote glede na radij okrog izhodišča
    mr5=((x[i]**2+y[i]**2+z[i]**2)/r0)**5      # obtežitev zaradi gostote glede na radij okrog izhodišča
    mr6=((x[i]**2+y[i]**2+z[i]**2)/r0)**8     # obtežitev zaradi gostote glede na radij okrog izhodišča

    rn=np.sqrt(x[i]**2+y[i]**2)

    if (x[i]**2+y[i]**2)<1 and (x[i]**2+z[i]**2)<1 and (z[i]**2+y[i]**2)<1:
        xN=np.append(xN,x[i])
        yN=np.append(yN,y[i])
        zN=np.append(zN,z[i])    
#       rN=np.append(rN,np.sqrt(x[i]**2+y[i]**2+z[i]**2))      # radij okrog izhodišča
        
        rN1=np.append(rN1, rn*mr1 )     # radij okrog osi
        rN2=np.append(rN2, rn*mr2 )     # radij okrog osi
        rN3=np.append(rN3, rn*mr3 )     # radij okrog osi
        rN4=np.append(rN4, rn*mr4 )     # radij okrog osi
        rN5=np.append(rN5, rn*mr5 )     # radij okrog osi
        rN6=np.append(rN6, rn*mr6 )     # radij okrog osi
        
    else :
        xZ=np.append(xZ,x[i])
        yZ=np.append(yZ,y[i])
        zZ=np.append(zZ,z[i])      
#        rZ=np.append(rZ,np.sqrt(x[i]**2+y[i]**2+z[i]**2))      # radij okrog izhodišča
#        rZ=np.append(rN1, rn*mr1 )    # radij okrog osi
        rZ1=np.append(rZ1, rn*mr1 )     # radij okrog osi
        rZ2=np.append(rZ2, rn*mr2 )     # radij okrog osi
        rZ3=np.append(rZ3, rn*mr3 )     # radij okrog osi
        rZ4=np.append(rZ4, rn*mr4 )     # radij okrog osi
        rZ5=np.append(rZ5, rn*mr5 )     # radij okrog osi
        rZ6=np.append(rZ6, rn*mr6 )


    if np.sqrt(x[i]**2+y[i]**2+z[i]**2)<1:   
        xkr=np.append(xkr,x[i])
        ykr=np.append(ykr,y[i])
        zkr=np.append(zkr,z[i])    
#        rkr=np.append(rkr,np.sqrt(x[i]**2+y[i]**2+z[i]**2))      # radij okrog izhodišča
        rkr1=np.append(rkr1, rn*mr1 )     # radij okrog osi
        rkr2=np.append(rkr2, rn*mr2 )     # radij okrog osi
        rkr3=np.append(rkr3, rn*mr3 )     # radij okrog osi
        rkr4=np.append(rkr4, rn*mr4 )     # radij okrog osi
        rkr5=np.append(rkr5, rn*mr5 )     # radij okrog osi
        rkr6=np.append(rkr6, rn*mr6 ) 
        
    if np.sqrt(x[i]**2+y[i]**2)<1:   
        xvaljOS=np.append(xvaljOS,x[i])
        yvaljOS=np.append(yvaljOS,y[i])
        zvaljOS=np.append(zvaljOS,z[i])    
#        rvaljOS=np.append(rvaljOS,np.sqrt(x[i]**2+y[i]**2+z[i]**2))      # radij okrog izhodišča
        rvaljOS1=np.append(rvaljOS1, rn*mr1 )  # radij okrog osi
        rvaljOS2=np.append(rvaljOS2, rn*mr2 )   # radij okrog osi
        rvaljOS3=np.append(rvaljOS3, rn*mr3 )   # radij okrog osi
        rvaljOS4=np.append(rvaljOS4, rn*mr4 )   # radij okrog osi
        rvaljOS5=np.append(rvaljOS5, rn*mr5 )   # radij okrog osi
        rvaljOS6=np.append(rvaljOS6, rn*mr6 )   # radij okrog osi
        
    if np.sqrt(x[i]**2+z[i]**2)<1:   
        xvaljPravok=np.append(xvaljPravok,x[i])
        yvaljPravok=np.append(yvaljPravok,y[i])
        zvaljPravok=np.append(zvaljPravok,z[i])    
#        rvaljPravok=np.append(rvaljPravok,np.sqrt(x[i]**2+y[i]**2+z[i]**2))      # radij okrog izhodišča
        rvaljPravok1=np.append(rvaljPravok1, rn*mr1 )   # radij okrog osi
        rvaljPravok2=np.append(rvaljPravok2, rn*mr2 )   # radij okrog osi
        rvaljPravok3=np.append(rvaljPravok3, rn*mr3 )   # radij okrog osi
        rvaljPravok4=np.append(rvaljPravok4, rn*mr4 )   # radij okrog osi
        rvaljPravok5=np.append(rvaljPravok5, rn*mr5 )   # radij okrog osi
        rvaljPravok6=np.append(rvaljPravok6, rn*mr6 )   # radij okrog osi
    
   
    if i==10**(m+1)-1:     #integrali oziroma vsota vseh vrednosti normaliziranega histograma => povpreča 
    
        Hkoc1, xkoc1, binEdge_koc1, Lkoc1=Hdata(np.append(rZ1**2,rN1**2))
        Hkoc2, xkoc2, binEdge_koc2, Lkoc2=Hdata(np.append(rZ2**2,rN2**2))
        Hkoc3, xkoc3, binEdge_koc3, Lkoc3=Hdata(np.append(rZ3**2,rN3**2))
        Hkoc4, xkoc4, binEdge_koc4, Lkoc4=Hdata(np.append(rZ4**2,rN4**2))
        Hkoc5, xkoc5, binEdge_koc5, Lkoc5=Hdata(np.append(rZ5**2,rN5**2))
        Hkoc6, xkoc6, binEdge_koc6, Lkoc6=Hdata(np.append(rZ6**2,rN6**2))

        Htelo1, xtelo1, binEdge_telo1, Ltelo1=Hdata(rN1**2)  
        Htelo2, xtelo2, binEdge_telo2, Ltelo2=Hdata(rN2**2)  
        Htelo3, xtelo3, binEdge_telo3, Ltelo3=Hdata(rN3**2)  
        Htelo4, xtelo4, binEdge_telo4, Ltelo4=Hdata(rN4**2)  
        Htelo5, xtelo5, binEdge_telo5, Ltelo5=Hdata(rN5**2)  
        Htelo6, xtelo6, binEdge_telo6, Ltelo6=Hdata(rN6**2)  
        
        HNEtelo1, xNEtelo1, binEdge_NEtelo1, LNEtelo1=Hdata(rZ1**2)  
        HNEtelo2, xNEtelo2, binEdge_NEtelo2, LNEtelo2=Hdata(rZ2**2)  
        HNEtelo3, xNEtelo3, binEdge_NEtelo3, LNEtelo3=Hdata(rZ3**2)  
        HNEtelo4, xNEtelo4, binEdge_NEtelo4, LNEtelo4=Hdata(rZ4**2)  
        HNEtelo5, xNEtelo5, binEdge_NEtelo5, LNEtelo5=Hdata(rZ5**2)  
        HNEtelo6, xNEtelo6, binEdge_NEtelo6, LNEtelo6=Hdata(rZ6**2)  

        Hkrog1, xkrog1, binEdge_krog1, Lkrog1=Hdata(rkr1**2)  
        Hkrog2, xkrog2, binEdge_krog2, Lkrog2=Hdata(rkr2**2)  
        Hkrog3, xkrog3, binEdge_krog3, Lkrog3=Hdata(rkr3**2)  
        Hkrog4, xkrog4, binEdge_krog4, Lkrog4=Hdata(rkr4**2)  
        Hkrog5, xkrog5, binEdge_krog5, Lkrog5=Hdata(rkr5**2)  
        Hkrog6, xkrog6, binEdge_krog6, Lkrog6=Hdata(rkr6**2)  

        HvaljOS1, x_valjOS1, binEdge_valjOS1, LvaljOS1=Hdata(rvaljOS1**2)  
        HvaljOS2, x_valjOS2, binEdge_valjOS2, LvaljOS2=Hdata(rvaljOS2**2)  
        HvaljOS3, x_valjOS3, binEdge_valjOS3, LvaljOS3=Hdata(rvaljOS3**2)  
        HvaljOS4, x_valjOS4, binEdge_valjOS4, LvaljOS4=Hdata(rvaljOS4**2)  
        HvaljOS5, x_valjOS5, binEdge_valjOS5, LvaljOS5=Hdata(rvaljOS5**2)  
        HvaljOS6, x_valjOS6, binEdge_valjOS6, LvaljOS6=Hdata(rvaljOS6**2)  

        HvaljPravok1, x_valjPravok1, binEdge_valjPravok1, LvaljPravok1=Hdata(rvaljPravok1**2)  
        HvaljPravok2, x_valjPravok2, binEdge_valjPravok2, LvaljPravok2=Hdata(rvaljPravok2**2)  
        HvaljPravok3, x_valjPravok3, binEdge_valjPravok3, LvaljPravok3=Hdata(rvaljPravok3**2)  
        HvaljPravok4, x_valjPravok4, binEdge_valjPravok4, LvaljPravok4=Hdata(rvaljPravok4**2)  
        HvaljPravok5, x_valjPravok5, binEdge_valjPravok5, LvaljPravok5=Hdata(rvaljPravok5**2)  
        HvaljPravok6, x_valjPravok6, binEdge_valjPravok6, LvaljPravok6=Hdata(rvaljPravok6**2)  

       
       
       
        momentKoc1=np.append(momentKoc1,np.sum(Hkoc1* np.diff(binEdge_koc1)* xkoc1 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKoc2=np.append(momentKoc2,np.sum(Hkoc2* np.diff(binEdge_koc2)* xkoc2 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKoc3=np.append(momentKoc3,np.sum(Hkoc3* np.diff(binEdge_koc3)* xkoc3 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKoc4=np.append(momentKoc4,np.sum(Hkoc4* np.diff(binEdge_koc4)* xkoc4 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKoc5=np.append(momentKoc5,np.sum(Hkoc5* np.diff(binEdge_koc5)* xkoc5 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKoc6=np.append(momentKoc6,np.sum(Hkoc6* np.diff(binEdge_koc6)* xkoc6 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov

        momentTelo1=np.append(momentTelo1,np.sum(Htelo1* np.diff(binEdge_telo1)* xtelo1 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentTelo2=np.append(momentTelo2,np.sum(Htelo2* np.diff(binEdge_telo2)* xtelo2 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentTelo3=np.append(momentTelo3,np.sum(Htelo3* np.diff(binEdge_telo3)* xtelo3 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentTelo4=np.append(momentTelo4,np.sum(Htelo4* np.diff(binEdge_telo4)* xtelo4 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentTelo5=np.append(momentTelo5,np.sum(Htelo5* np.diff(binEdge_telo5)* xtelo5 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentTelo6=np.append(momentTelo6,np.sum(Htelo6* np.diff(binEdge_telo6)* xtelo6 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov

        momentNETelo1=np.append(momentNETelo1,np.sum(HNEtelo1* np.diff(binEdge_NEtelo1)* xNEtelo1 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentNETelo2=np.append(momentNETelo2,np.sum(HNEtelo2* np.diff(binEdge_NEtelo2)* xNEtelo2 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentNETelo3=np.append(momentNETelo3,np.sum(HNEtelo3* np.diff(binEdge_NEtelo3)* xNEtelo3 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentNETelo4=np.append(momentNETelo4,np.sum(HNEtelo4* np.diff(binEdge_NEtelo4)* xNEtelo4 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentNETelo5=np.append(momentNETelo5,np.sum(HNEtelo5* np.diff(binEdge_NEtelo5)* xNEtelo5 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov
        momentNETelo6=np.append(momentNETelo6,np.sum(HNEtelo6* np.diff(binEdge_NEtelo6)* xNEtelo6 ) ) #np.diff vrne razliko med vrednostmi sosednjih elementov

        momentKrog1=np.append(momentKrog1,np.sum(Hkrog1* np.diff(binEdge_krog1)* xkrog1 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKrog2=np.append(momentKrog2,np.sum(Hkrog2* np.diff(binEdge_krog2)* xkrog2 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKrog3=np.append(momentKrog3,np.sum(Hkrog3* np.diff(binEdge_krog3)* xkrog3 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKrog4=np.append(momentKrog4,np.sum(Hkrog4* np.diff(binEdge_krog4)* xkrog4 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKrog5=np.append(momentKrog5,np.sum(Hkrog5* np.diff(binEdge_krog5)* xkrog5 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentKrog6=np.append(momentKrog6,np.sum(Hkrog6* np.diff(binEdge_krog6)* xkrog6 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov

        momentvaljOS1=np.append(momentvaljOS1,np.sum(HvaljOS1* np.diff(binEdge_valjOS1)* x_valjOS1 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljOS2=np.append(momentvaljOS2,np.sum(HvaljOS2* np.diff(binEdge_valjOS2)* x_valjOS2 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljOS3=np.append(momentvaljOS3,np.sum(HvaljOS3* np.diff(binEdge_valjOS3)* x_valjOS3 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljOS4=np.append(momentvaljOS4,np.sum(HvaljOS4* np.diff(binEdge_valjOS4)* x_valjOS4 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljOS5=np.append(momentvaljOS5,np.sum(HvaljOS5* np.diff(binEdge_valjOS5)* x_valjOS5 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljOS6=np.append(momentvaljOS6,np.sum(HvaljOS6* np.diff(binEdge_valjOS6)* x_valjOS6 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov

        momentvaljPravok1=np.append(momentvaljPravok1,np.sum(HvaljPravok1* np.diff(binEdge_valjPravok1)* x_valjPravok1 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljPravok2=np.append(momentvaljPravok2,np.sum(HvaljPravok2* np.diff(binEdge_valjPravok2)* x_valjPravok2 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljPravok3=np.append(momentvaljPravok3,np.sum(HvaljPravok3* np.diff(binEdge_valjPravok3)* x_valjPravok3 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljPravok4=np.append(momentvaljPravok4,np.sum(HvaljPravok4* np.diff(binEdge_valjPravok4)* x_valjPravok4 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljPravok5=np.append(momentvaljPravok5,np.sum(HvaljPravok5* np.diff(binEdge_valjPravok5)* x_valjPravok5 ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        momentvaljPravok6=np.append(momentvaljPravok6,np.sum(HvaljPravok6* np.diff(binEdge_valjPravok6)* x_valjPravok6) )#np.diff vrne razliko med vrednostmi sosednjih elementov

        print(m)
        n[m]=i
        m=m+1
        


###### ____izris_grafov_in_histogramov______________________________

F10=plt.figure(10)
plt.suptitle(r'Normirani histogrami porazdelitve momenta $r_z^2$ za N='+str(N)+' točk',fontsize=16)
F10=plt.subplot(2, 3, 1 ) 
plt.step(xkoc1,Hkoc1,'y:',alpha = 0.9,label=r'p=0.2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKoc1[m-1], 5 ))
plt.step(xkoc3,Hkoc3,'y-',alpha = 0.5,label=r'p=1  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKoc3[m-1], 5 ))
plt.step(xkoc4,Hkoc4,'y',alpha = 0.2,label=r'p=2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKoc4[m-1], 5 ))
plt.xlim(0,4) 
plt.xlabel(r'$ r_z^2 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.title('Točke znotraj kocke prvega kvadranta')#+'(N='+str(N)+',M='+str(M)+')')
F10=plt.subplot(2, 3, 2 ) 
plt.step(xtelo1,Htelo1,'r:',alpha = 0.9,label=r'p=0.2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentTelo1[m-1], 5 ))
plt.step(xtelo3,Htelo3,'r-',alpha = 0.5,label=r'p=1  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentTelo2[m-1], 5 ))
plt.step(xtelo4,Htelo4,'r',alpha = 0.2,label=r'p=2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentTelo3[m-1], 5 ))
plt.xlim(0,1.5) 
plt.xlabel(r'$ r_z^2 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.title('Točke znotraj preseka treh valjev')#+'(N='+str(N)+',M='+str(M)+')')
F10=plt.subplot(2, 3, 3 ) 
plt.step(xNEtelo1,HNEtelo1,'r:',alpha = 0.9,label=r'p=0.2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentNETelo1[m-1], 5 ))
plt.step(xNEtelo3,HNEtelo3,'r-',alpha = 0.5,label=r'p=1  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentNETelo3[m-1], 5 ))
plt.step(xNEtelo4,HNEtelo4,'r',alpha = 0.2,label=r'p=2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentNETelo4[m-1], 5 ))
plt.xlim(0,10) 
plt.xlabel(r'$ r_z^2 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.title('Točke zunaj preseka treh valjev')#+'(N='+str(N)+',M='+str(M)+')')
F10=plt.subplot(2, 3, 4 ) 
plt.step(xkrog1,Hkrog1,'b:',alpha = 0.9,label=r'p=0.2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKrog1[m-1], 5 ))
plt.step(xkrog3,Hkrog3,'b-',alpha = 0.5,label=r'p=1  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKrog3[m-1], 5 ))
plt.step(xkrog4,Hkrog4,'b',alpha = 0.2,label=r'p=2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentKrog4[m-1], 5 ))
plt.xlim(0,1) 
plt.xlabel(r'$ r_z^2 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.title('Točke znotraj radija r<1')#+'(N='+str(N)+',M='+str(M)+')')
F10=plt.subplot(2, 3, 5 ) 
plt.step(x_valjOS1,HvaljOS1,'b:',alpha = 0.9,label=r'p=0.2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljOS1[m-1], 5 ))
plt.step(x_valjOS3,HvaljOS3,'b-',alpha = 0.5,label=r'p=1  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljOS3[m-1], 5 ))
plt.step(x_valjOS4,HvaljOS4,'b',alpha = 0.2,label=r'p=2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljOS4[m-1], 5 ))
plt.xlim(0,2) 
plt.xlabel(r'$ r_z^2 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.title('Točke pokoncnega valja')#+'(N='+str(N)+',M='+str(M)+')')
F10=plt.subplot(2, 3, 6 ) 
plt.step(x_valjPravok1,HvaljPravok1,'b:',alpha = 0.9,label=r'p=0.2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljPravok1[m-1], 5 ))
plt.step(x_valjPravok3,HvaljPravok3,'b-',alpha = 0.5,label=r'p=1  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljPravok3[m-1], 5 ))
plt.step(x_valjPravok4,HvaljPravok4,'b',alpha = 0.2,label=r'p=2  $\langle r_z^2 \rangle=$'+'{:.{}f}'.format(momentvaljPravok4[m-1], 5 ))
plt.xlim(0,4) 
plt.xlabel(r'$ r_z^2 $')
plt.ylabel('n')
plt.legend(loc='best')
plt.title('Točke ležečega valja')#+'(N='+str(N)+',M='+str(M)+')')

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
#plt.title('Točke prvega kvadranta znotraj preseka treh valjev')#+'(N='+str(N)+',M='+str(M)+')')
##F30=plt.subplot(1, 3, 2 ) 
#F30=plt.figure(30)
#Axes3D = plt.axes(projection='3d')
#Axes3D.scatter(xZ, yZ, zZ, zdir='z', s=1, c='c', alpha = 0.1, depthshade=False)
#plt.title('Točke prvega kvadranta zunaj preseka treh valjev')#+'(N='+str(N)+',M='+str(M)+')')
##F40=plt.subplot(1, 3, 3 ) 
#F40=plt.figure(40)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xZ, yZ, zZ, zdir='z', s=1, c='c', alpha = 0.1, depthshade=False)
#Axes3D.scatter(xN, yN, zN, zdir='z', s=1, c='r',alpha = 0.1, depthshade=False)
##plt.title('Točke znotraj radija r<1')#+'(N='+str(N)+',M='+str(M)+')')
#plt.title('Točke prvega kvadranta')#+'(N='+str(N)+',M='+str(M)+')')
#
#F41=plt.figure(41)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xvaljOS, yvaljOS, zvaljOS, zdir='z', s=1, c='g', alpha = 0.1, depthshade=False)
#plt.title('Točke pokončnega valja')#+'(N='+str(N)+',M='+str(M)+')')
#
#F42=plt.figure(42)
#Axes3D = plt.axes(projection='3d')
##Axes3D.scatter(xkr, ykr, zkr, zdir='z', s=1, c='b', alpha = 0.1, depthshade=True)
#Axes3D.scatter(xvaljPravok, yvaljPravok, zvaljPravok, zdir='z', s=1, c='k', alpha = 0.1, depthshade=False)
#plt.title('Točke ležečega valja')#+'(N='+str(N)+',M='+str(M)+')')

#
##
#
F50=plt.figure(50)

F50=plt.subplot(2, 1, 1 ) 
plt.plot(n,abs(momentKoc3),'yo',alpha = 0.95,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo3),'ro',alpha = 0.95,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentNETelo3),'co',alpha = 0.95,label=r'$zunaj\ preseka$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentKrog3),'bo',alpha = 0.95,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentvaljOS3),'go',alpha = 0.95,label=r'$pokončni\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentvaljPravok3),'ko',alpha = 0.95,label=r'$ležeči\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xscale('log')
#plt.yscale('log')
plt.xlabel(' N ')
plt.ylabel(r'$\langle r_z^2 \rangle$')
plt.title('Absolutne vrednosti momentov pri $p=1$ za različno velike vzorce')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

F50=plt.subplot(2, 1, 2 )  
#plt.plot(abs(np.diff(momentKoc)),'y',alpha = 0.5,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(np.diff(momentTelo)),'r',alpha = 0.5,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(np.diff(momentKrog)),'b',alpha = 0.5,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentKoc2-momentKoc2[potenca-1]),'yo',alpha = 0.95,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo2-momentTelo2[potenca-1]),'ro',alpha = 0.95,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentNETelo2-momentNETelo2[potenca-1]),'co',alpha = 0.95,label=r'$zunaj\ preseka$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentKrog2-momentKrog2[potenca-1]),'bo',alpha = 0.95,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentvaljOS2-momentvaljOS2[potenca-1]),'go',alpha = 0.95,label=r'$pokončni\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentvaljPravok2-momentvaljPravok2[potenca-1]),'ko',alpha = 0.95,label=r'$ležeči\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(' N ')
plt.ylabel(r'$\langle r_z^2 \rangle_{N_i}-\langle r_z^2 \rangle_{N=10^6}$')
plt.title('Absolutno odstopanje izračuna momenta pri $p=1$ od točne vrednosti za vsako povečanje velikosti vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)




p=np.array([0.2,0.5,1,2,5,8])
momentKoc=np.array([momentKoc1[potenca-1],momentKoc2[potenca-1],momentKoc3[potenca-1],momentKoc4[potenca-1],momentKoc5[potenca-1],momentKoc6[potenca-1]])
momentTelo=np.array([momentTelo1[potenca-1],momentTelo2[potenca-1],momentTelo3[potenca-1],momentTelo4[potenca-1],momentTelo5[potenca-1],momentTelo6[potenca-1]])
momentNETelo=np.array([momentNETelo1[potenca-1],momentNETelo2[potenca-1],momentNETelo3[potenca-1],momentNETelo4[potenca-1],momentNETelo5[potenca-1],momentNETelo6[potenca-1]])
momentKrog=np.array([momentKrog1[potenca-1],momentKrog2[potenca-1],momentKrog3[potenca-1],momentKrog4[potenca-1],momentKrog5[potenca-1],momentKrog6[potenca-1],])
momentvaljOS=np.array([momentvaljOS1[potenca-1],momentvaljOS2[potenca-1],momentvaljOS3[potenca-1],momentvaljOS4[potenca-1],momentvaljOS5[potenca-1],momentvaljOS6[potenca-1],])
momentvaljPravok=np.array([momentvaljPravok1[potenca-1],momentvaljPravok2[potenca-1],momentvaljPravok3[potenca-1],momentvaljPravok4[potenca-1],momentvaljPravok5[potenca-1],momentvaljPravok6[potenca-1]])


F60=plt.figure(60)

F60=plt.subplot(3, 1, 1 ) 
plt.plot(n,abs(momentTelo1),'yo',alpha = 0.95,label=r'$p=0.2$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo2),'ro',alpha = 0.95,label=r'$p=0.5$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo3),'co',alpha = 0.95,label=r'$p=1$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo4),'bo',alpha = 0.95,label=r'$p=2$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo5),'go',alpha = 0.95,label=r'$p=5$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo6),'ko',alpha = 0.95,label=r'$p=8$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xscale('log')
#plt.yscale('log')
plt.ylabel(r'$\langle r_z^2 \rangle$')
plt.xlabel(' N ')
plt.title('Absolutne vrednosti momentov preseka valjev za različne vrednosti $p$')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

F60=plt.subplot(3, 1, 2 )  
#plt.plot(abs(np.diff(momentKoc)),'y',alpha = 0.5,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(np.diff(momentTelo)),'r',alpha = 0.5,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(np.diff(momentKrog)),'b',alpha = 0.5,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo1-momentTelo1[potenca-1]),'yo',alpha = 0.95,label=r'$p=0.2$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo2-momentTelo2[potenca-1]),'ro',alpha = 0.95,label=r'$p=0.5$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo3-momentTelo3[potenca-1]),'co',alpha = 0.95,label=r'$p=1$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo4-momentTelo4[potenca-1]),'bo',alpha = 0.95,label=r'$p=2$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo5-momentTelo5[potenca-1]),'go',alpha = 0.95,label=r'$p=5$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(n,abs(momentTelo6-momentTelo6[potenca-1]),'ko',alpha = 0.95,label=r'$p=8$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(' N ')
plt.ylabel(r'$\langle r_z^2 \rangle_{N_i}-\langle r_z^2 \rangle_{N=10^6}$')
plt.title('Absolutno odstopanje izračuna momenta preseka valjev od vrednosti dobljenih pri $N=$'+str(N)+ ' za različne vrednosti $p$' )#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=2)


F60=plt.subplot(3, 1, 3 )  
#plt.plot(abs(np.diff(momentKoc)),'y',alpha = 0.5,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(np.diff(momentTelo)),'r',alpha = 0.5,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(np.diff(momentKrog)),'b',alpha = 0.5,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(p,momentKoc,'yo',alpha = 0.95,label=r'$kocka$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(p,momentTelo,'ro',alpha = 0.95,label=r'$presek\ valjev$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(p,momentNETelo,'co',alpha = 0.95,label=r'$zunaj\ preseka$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(p,momentKrog,'bo',alpha = 0.95,label=r'$krogla$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(p,momentvaljOS,'go',alpha = 0.95,label=r'$pokončni\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(p,momentvaljPravok,'ko',alpha = 0.95,label=r'$ležeči\ valj$')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r' $p$ ')
plt.ylabel(r'$\langle r_z^2 \rangle$')
plt.title('Momenti v odvisnosti od vrednosti parametra $p$ računani pri $N=$'+str(N))#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=2)

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
