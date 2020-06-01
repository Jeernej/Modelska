# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:15:15 2019

@author: jernej
"""
import numpy as np
import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize

from numpy import diff

    
e=2.718281828459045    
pi=3.141592653589793 

##    
#def VV(zacetni):
#    dx=1/50
#    return sum(zacetni*dx)-1

#def FFun1(v,vz,vmax,lambdav1):
#    delta_t=1/20
#    vsum=((v[0]-vz)/delta_t)**2
#    expsum=vz/2
##    expmax=e**(vz-vmax)
##    expmin=m.exp(vmin-vz)
#    delilne_tocke=len(v)
#    for i in range(0,delilne_tocke-1,1):
#        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
#    for i in range(0,delilne_tocke,1):
#        if i<(delilne_tocke-1):
#            expsum=expsum+v[i]
#        else:
#            expsum=expsum+v[i]/2
#    for i in range(0,delilne_tocke,1):
#        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
#        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
#    return (vsum+1*e**(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*e**(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t))))
#   
def fun1(x,yz):   
    return 3*(x-x**2/2)*(1-yz)+yz   ## prosti končni robni pogoj

def Fun1(zacetni,yz,lamb):  ## prosti končni robni pogoj
#    tol=10**(-5)
    l0=1
    vmin=0.9
    vmax=1.2
    zacetni[0]=yz
    
#    LagMulti=zacetni[-1]  ## lagrangejev multiplikator spravljen kot zadnji element vektorja
    
    I_v=sum(zacetni[0:-1])#+yz/2+zacetni[-2]/2   ## integral hitrosti začetna hitrost=yz je fiksna 
    I_v=I_v/(len(zacetni)-1)                    ### normalizirano glede na število točk v intervalu 

    dv=diff(zacetni[0:-1])      ## izračuna razlike za odvod hitrosti - pospešek
#    dv[0]=dv[0]/2
#    dv[-1]=dv[-1]/2
    I_dv2=sum((dv*(len(zacetni)-1))**2)          ## integral hvadrata odvoda hitrosti - pospeška
    I_dv2=I_dv2/(len(zacetni)-1)                ### normalizirano glede na število točk v intervalu 
    
#    dx=1/(len(zacetni)-1)
#    exp1=sum(e**(zacetni[1:-2]*dx) + yz/2*dx + zacetni[-2]/2*dx -l0)       ## kaznuj odstopanje od  integral(v*dt)=pot
#    exp2=sum(e**(zacetni[1:-2]*dx) - yz/2*dx - zacetni[-2]/2*dx +l0-tol)  ## kaznuj odstopanje od   integral(v*dt)=pot
    
    V_MAX=0
    V_MIN=0
    for i in zacetni[0:-1]:
        V_MIN=V_MIN+(e**(lamb*( vmin-i)))    ## kaznuje premajhne (negativne) hitrosti
        V_MAX=V_MAX+(e**(lamb*(-vmax+i)))    ## kaznuj prevelike hitrosti
#        if i<vmin:
#            V_MIN=V_MIN+(e**(lamb*( vmin-i)))    ## kaznuje premajhne (negativne) hitrosti
#        if i>vmax:
#            V_MAX=V_MAX+(e**(lamb*(-vmax+i)))    ## kaznuj prevelike hitrosti
            
#    lag=0
#    if LagMulti<6*(1-yz)-6 or LagMulti>6*(1-yz)+10: lag=e**(LagMulti) ## kaznuj prevelik lagrangejev multiplikator
#        
    return I_dv2 - I_v  +  V_MIN + V_MAX + e**(10*abs(I_v-l0))# + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot
#    return I_dv2 - I_v * LagMulti +  V_MIN + V_MAX + e**(9*(I_v-l0+tol))+e**(9*(-I_v+l0-tol)) + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot



def fun2(x,yz,yk):    
    lamb=12*(2-yz-yk)
    A=lamb/4+yk-yz
    B=yz
    return -lamb/4*x**2+A*x+B  ### omejitev končne hitrosti
    
def Fun2(zacetni,yz,yk,lamb):  ## prosti končni robni pogoj
#    tol=10**(-5)
    l0=1
    vmin=0.9
    vmax=1.2
    zacetni[0]=yz
    zacetni[-2]=yk

    
#    LagMulti=zacetni[-1]  ## lagrangejev multiplikator spravljen kot zadnji element vektorja
    
    I_v=sum(zacetni[0:-1])#+yz/2+zacetni[-2]/2   ## integral hitrosti začetna hitrost=yz je fiksna 
    I_v=I_v/(len(zacetni)-1)                    ### normalizirano glede na število točk v intervalu 

    dv=diff(zacetni[0:-1])      ## izračuna razlike za odvod hitrosti - pospešek
#    dv[0]=dv[0]/2
#    dv[-1]=dv[-1]/2
    I_dv2=sum((dv*(len(zacetni)-1))**2)          ## integral hvadrata odvoda hitrosti - pospeška
    I_dv2=I_dv2/(len(zacetni)-1)                ### normalizirano glede na število točk v intervalu 
    
#    dx=1/(len(zacetni)-1)
#    exp1=sum(e**(zacetni[1:-2]*dx) + yz/2*dx + zacetni[-2]/2*dx -l0)       ## kaznuj odstopanje od  integral(v*dt)=pot
#    exp2=sum(e**(zacetni[1:-2]*dx) - yz/2*dx - zacetni[-2]/2*dx +l0-tol)  ## kaznuj odstopanje od   integral(v*dt)=pot
    
    V_MAX=0
    V_MIN=0
    for i in zacetni[0:-1]:
        V_MIN=V_MIN+(e**(lamb*( vmin-i)))    ## kaznuje premajhne (negativne) hitrosti
        V_MAX=V_MAX+(e**(lamb*(-vmax+i)))    ## kaznuje prevelike hitrosti
#        if i<vmin:
#            V_MIN=V_MIN+(e**(lamb*( vmin-i)))    ## kaznuje premajhne (negativne) hitrosti
#        if i>vmax:
#            V_MAX=V_MAX+(e**(lamb*(-vmax+i)))    ## kaznuje prevelike hitrosti
            
#    lag=0
#    if LagMulti<6*(1-yz)-6 or LagMulti>6*(1-yz)+10: lag=e**(LagMulti) ## kaznuj prevelik lagrangejev multiplikator
#        
    return I_dv2 - I_v  +  V_MIN + V_MAX + e**(11*abs(I_v-l0))# + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot
#    return I_dv2 - I_v * LagMulti +  V_MIN + V_MAX + e**(9*(I_v-l0+tol))+e**(9*(-I_v+l0-tol)) + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot


    
def fun3(x,yz,p):  
    return (4*p-1)*(1-yz)*( 1 - (1-x)**(2*p/(2*p-1)) ) / (2*p-1) + yz  ### prosti robni pogoj pri potenci


def Fun3(zacetni,yz,p):  ## prosti končni robni pogoj
#    tol=10**(-5)
    l0=1
    vmin=0.
    vmax=5
    zacetni[0]=yz
    
#    LagMulti=zacetni[-1]  ## lagrangejev multiplikator spravljen kot zadnji element vektorja
    
    I_v=sum(zacetni[0:-1])#+yz/2+zacetni[-2]/2   ## integral hitrosti začetna hitrost=yz je fiksna 
    I_v=I_v/(len(zacetni)-1)                    ### normalizirano glede na število točk v intervalu 

    dv=diff(zacetni[0:-1])      ## izračuna razlike za odvod hitrosti - pospešek
#    dv[0]=dv[0]/2
#    dv[-1]=dv[-1]/2
    I_dv2=sum((dv*(len(zacetni)-1))**(2*p))          ## integral hvadrata odvoda hitrosti - pospeška
    I_dv2=I_dv2/(len(zacetni)-1)                ### normalizirano glede na število točk v intervalu 
    
#    dx=1/(len(zacetni)-1)
#    exp1=sum(e**(zacetni[1:-2]*dx) + yz/2*dx + zacetni[-2]/2*dx -l0)       ## kaznuj odstopanje od  integral(v*dt)=pot
#    exp2=sum(e**(zacetni[1:-2]*dx) - yz/2*dx - zacetni[-2]/2*dx +l0-tol)  ## kaznuj odstopanje od   integral(v*dt)=pot
    
    V_MAX=0
    V_MIN=0
    for i in zacetni[0:-1]:
        if i<vmin:
            V_MIN=V_MIN+(e**(9*( vmin-i)))    ## kaznuje premajhne (negativne) hitrosti
        if i>vmax:
            V_MAX=V_MAX+(e**(9*(-vmax+i)))    ## kaznuje prevelike hitrosti
            
#    lag=0
#    if LagMulti<6*(1-yz)-6 or LagMulti>6*(1-yz)+10: lag=e**(LagMulti) ## kaznuj prevelik lagrangejev multiplikator
#        
    return I_dv2 - I_v  +  V_MIN + V_MAX + e**(10*abs(I_v-l0))# + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot
#    return I_dv2 - I_v * LagMulti +  V_MIN + V_MAX + e**(9*(I_v-l0+tol))+e**(9*(-I_v+l0-tol)) + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot

    
def fun33(x,yz):  
    return 2*x*(1-yz)+yz  ### pogoj pri potenci 'p=infty'
    
    
    
def fun4(x,yz,mu):   ## prosti robni pogoj   +  Kvadrat hitrosti v funkcionalu
    return (mu - yz * np.tanh(mu)) / (mu - np.tanh(mu)) + (yz - (mu - yz * np.tanh(mu)) / (mu - np.tanh(mu)) )  * (e**(mu*x) + e**(2*mu-mu*x)) / (1 + e**(2*mu) )

def Fun4(zacetni,yz,mu):  ## prosti končni robni pogoj
#    tol=10**(-5)
    l0=1
    vmin=0.
    vmax=2
    zacetni[0]=yz
    
#    LagMulti=zacetni[-1]  ## lagrangejev multiplikator spravljen kot zadnji element vektorja
    
    I_v=sum(zacetni[0:-1])#+yz/2+zacetni[-2]/2   ## integral hitrosti začetna hitrost=yz je fiksna 
    I_v=I_v/(len(zacetni)-1)                    ### normalizirano glede na število točk v intervalu 

    dv=diff(zacetni[0:-1])      ## izračuna razlike za odvod hitrosti - pospešek
#    dv[0]=dv[0]/2
#    dv[-1]=dv[-1]/2
    I_dv2=sum((dv*(len(zacetni)-1))**(2))          ## integral hvadrata odvoda hitrosti - pospeška
    I_dv2=I_dv2/(len(zacetni)-1)                ### normalizirano glede na število točk v intervalu 
    
#    dx=1/(len(zacetni)-1)
#    exp1=sum(e**(zacetni[1:-2]*dx) + yz/2*dx + zacetni[-2]/2*dx -l0)       ## kaznuj odstopanje od  integral(v*dt)=pot
#    exp2=sum(e**(zacetni[1:-2]*dx) - yz/2*dx - zacetni[-2]/2*dx +l0-tol)  ## kaznuj odstopanje od   integral(v*dt)=pot
    
    V_MAX=0
    V_MIN=0
    for i in zacetni[0:-1]:
        if i<vmin:
            V_MIN=V_MIN+(e**(9*( vmin-i)))    ## kaznuje premajhne (negativne) hitrosti
        if i>vmax:
            V_MAX=V_MAX+(e**(9*(-vmax+i)))    ## kaznuj prevelike hitrosti
            
#    lag=0
#    if LagMulti<6*(1-yz)-6 or LagMulti>6*(1-yz)+10: lag=e**(LagMulti) ## kaznuj prevelik lagrangejev multiplikator
#        
    return I_dv2 + (mu*I_v)**2 - I_v  +  V_MIN + V_MAX + e**(10*abs(I_v-l0))# + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot
#    return I_dv2 - I_v * LagMulti +  V_MIN + V_MAX + e**(9*(I_v-l0+tol))+e**(9*(-I_v+l0-tol)) + lag  ## zadnja člena kanujeta odstopanje od: integral(v*dt)=pot



def fun5(x,yz):  
    return 6*x*(1-x)*(1-yz)+yz ## periodični robni pogoj  (verzija  omejitev končne hitrosti)




###### ___zaćetne rednosti

#x=np.linspace(0,1,21)#

###### ___minimizacija


#res1 = minimize(energija, zacetni, method='Nelder-Mead', tol=1e-9, options={'maxiter': 5000}) ## 'Nelder-Mead 
#print(res1)
#res2 = minimize(energija, zacetni, method='Powell', tol=1e-9, options={'maxiter': 5000}) ## 'Nelder-Mead 
#print(res2)
#res3 = minimize(lagrangian, zacetni, method='L-BFGS-B', tol=1e-9, options={'maxiter': 5000}) ## 'Nelder-Mead 
#print(res3)

#Rl_Pot_En_min1=energija(res1.x)
#Rl_Pot_En_min2=energija(res2.x)
#Rl_Pot_En_min3=energija(res3.x)

#print(Rl_Pot_En_min1)
#print(Rl_Pot_En_min2)
#print(Rl_Pot_En_min3)


###### ____rezultatu_v _vektorjih________________________

barva=['r','b','g','y','m','c','o']
LS=['-','--',':']
MARK=['x','o','<']

x=np.linspace(0,1,50)#-len(FTSIG)/2

lamb=0.1
#lamb=1
#lamb=5

F60=plt.figure(60)
#F60=plt.subplot(1, 1, 1 ) 
plt.title(r'Prosti robni pogoj pri $v_{min}$ in $v_{max}$ za $\lambda=$'+str(lamb),fontsize=16)

c=0
for yz in [0.,0.5,1.,1.5]: # različne začetne hitrosti
#    F60=plt.subplot(2, 1, 1 ) 
#    plt.plot(x,fun1(x,yz), color=barva[c], LS='-', alpha = 0.75,  label=r'$y_0$={}'.format(yz))  ## analitična rešitev za prosti robni pogoj
#    res1 = minimize(Fun1, x, args=yz, method='L-BFGS-B')
#    plt.plot(x,res1.x, color=barva[c], LS=':',alpha = 0.95,  label=r'$y_0$={}'.format(yz))
    lagMult=6*(1-yz) ## začetni približek za lagrangejev multiplikator
    res11 = minimize(Fun1, np.append(x,lagMult), args=(yz,lamb), method='L-BFGS-B',tol=1e-10,options={'maxiter': 5000})   ## minimizacija za prosti robni pogoj
#    res11 = minimize(Fun1, np.append(x,lagMult), args=(yz), method='Powell',tol=1e-6)   ## minimizacija za prosti robni pogoj
    print('success='+ str(res11.success))
#    res11.x[0]=yz ## popravim začetni pogoj
    plt.plot(x,res11.x[0:-1], color=barva[c], LS='-',alpha = 0.95,  label=r'$y_n$={}'.format(yz))
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y(x)$',fontsize=16)
    plt.xlim([0,1.5])

    plt.legend(loc='best')

#    F60=plt.subplot(2, 1, 2 ) 
#    plt.plot(x,abs(res11.x[0:-1]-fun1(x,yz)), color=barva[c], LS='-',alpha = 0.75,  label=r'$y_0$={}'.format(yz))

    c=c+1
    
#plt.xlabel(r'$x$',fontsize=16)
#plt.ylabel(r'$|y_{a}(x)-y_{n}(x)|$',fontsize=16)
#plt.xlim([0,1.2])
#plt.yscale('log')
#plt.legend(loc='best')

################################################################################
#
F660=plt.figure(660)
#F60=plt.subplot(1, 1, 1 ) 
plt.title(r'Periodični robni pogoj pri $v_{min}$ in $v_{max}$ za $\lambda=$'+str(lamb),fontsize=16)
x=np.linspace(0,1,15)#-len(FTSIG)/2

c=0
for yz in [0.,0.5,1.,1.5]: # števila polov
#    F660=plt.subplot(2, 1, 1 ) 
#    plt.plot(x,fun5(x,yz), color=barva[c], alpha = 0.95,  label=r'$y_0$={}'.format(yz))
#    res1 = minimize(Fun1, x, args=yz, method='L-BFGS-B')
#    plt.plot(x,res1.x, color=barva[c], LS=':',alpha = 0.95,  label=r'$y_0$={}'.format(yz))
    lagMult=6*(1-yz) ## začetni približek za lagrangejev multiplikator
    res5 = minimize(Fun2, np.append(x,lagMult), args=(yz,yz,lamb), method='L-BFGS-B',tol=1e-10,options={'maxiter': 5000})   ## minimizacija za prosti robni pogoj
#    res11 = minimize(Fun1, np.append(x,lagMult), args=(yz), method='Powell',tol=1e-6)   ## minimizacija za prosti robni pogoj
    print('success='+ str(res5.success))
#    res11.x[0]=yz ## popravim začetni pogoj
    plt.plot(x,res5.x[0:-1], color=barva[c], LS='-',alpha = 0.95,  label=r'$y_n$=$y_T$={}'.format(yz))
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y(x)$',fontsize=16)
    plt.xlim([0,1.5])
    plt.legend(loc='best')

#    F660=plt.subplot(2, 1, 2 ) 
#    plt.plot(x,abs(res5.x[0:-1]-fun5(x,yz)), color=barva[c], LS='-',alpha = 0.75,  label=r'$y_0$={}'.format(yz))
    c=c+1
#    
#plt.xlabel(r'$x$',fontsize=16)
#plt.ylabel(r'$|y_{a}(x)-y_{n}(x)|$',fontsize=16)
#plt.xlim([0,1.2])
#plt.yscale('log')
#plt.legend(loc='best')


#################################################################################


F10=plt.figure(10)
plt.title(r'Omejitev končne hitrosti pri $v_{min}$ in $v_{max}$ za $\lambda=$'+str(lamb),fontsize=16)
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1.5])
x=np.linspace(0,1,50)#-len(FTSIG)/2

d=0
for yk in [0.5,1,1.5]: # števila polov
    c=0
    for yz in [0.5,1.,1.5]: # števila polov
#        F10=plt.subplot(2, 1, 1 ) 
#        plt.plot(x,fun2(x,yz,yk), color=barva[c],ls=LS[d], alpha = 0.75, label=r'$y_0$={}'.format(yz)+r',  $y_T$={}'.format(yk))
        lagMult=6*(1-yz) ## začetni približek za lagrangejev multiplikator
        res2 = minimize(Fun2, np.append(x,lagMult), args=(yz,yk,lamb), method='L-BFGS-B',tol=1e-10,options={'maxiter': 5000})   ## minimizacija za prosti robni pogoj
#        plt.plot(x,res2.x, color=barva[c],ls=':', alpha = 0.95, label=r'$y_0$={}'.format(yz)+r',  $y_T$={}'.format(yk))
        print('success='+ str(res11.success))
    #    res11.x[0]=yz ## popravim začetni pogoj
        plt.plot(x,res2.x[0:-1], color=barva[c], LS=LS[d],alpha = 0.65, label=r'$y_0$={}'.format(yz)+r',  $y_T$={}'.format(yk))
        plt.xlabel(r'$x$',fontsize=16)
        plt.ylabel(r'$y(x)$',fontsize=16)
        plt.xlim([0,1.5])
    
        plt.legend(loc='best')
    
#        F10=plt.subplot(2, 1, 2 ) 
#        plt.plot(x,abs(res2.x[0:-1]-fun2(x,yz,yk)), color=barva[c],ls=LS[d] ,alpha = 0.75,   label=r'$y_0$={}'.format(yz)+r',  $y_T$={}'.format(yk))
        c=c+1
    d=d+1

#plt.xlabel(r'$x$',fontsize=16)
#plt.ylabel(r'$|y_{a}(x)-y_{n}(x)|$',fontsize=16)
#plt.xlim([0,1.2])
#plt.yscale('log')
#plt.legend(loc='best')



#    
