# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
import matplotlib.pyplot  as plt

e=2.718281828459045    
pi=3.141592653589793 

barva=['r','b','g','y','m','c','o']
LS=['-','--',':']


def fun1(x,yz):  
    return 3*(x-x**2/2)*(1-yz)+yz


def fun2(x,yz,yk):  
    lamb=12*(2-yz-yk)
    A=lamb/4+yk-yz
    B=yz
    return -lamb/4*x**2+A*x+B
    
    
    
def fun3(x,yz,p):  
    return (4*p-1)*(1-yz)*( 1 - (1-x)**(2*p/(2*p-1)) ) / (2*p-1) + yz
    
def fun33(x,yz):  
    return 2*x*(1-yz)+yz
    
    
    
def fun4(x,yz,p):  
    return (mu - yz * np.tanh(mu)) / (mu - np.tanh(mu)) + (yz - (mu - yz * np.tanh(mu)) / (mu - np.tanh(mu)) )  * (e**(mu*x) + e**(2*mu-mu*x)) / (1 + e**(2*mu) )



def fun5(x,yz):  
    return 6*x*(1-x)*(1-yz)+yz


##################################################################################

F60=plt.figure(60)
#F60=plt.subplot(1, 1, 1 ) 
plt.title('Vožnja do semaforja za prosti robni pogoj',fontsize=16)
x=np.linspace(0,1,100)#-len(FTSIG)/2

c=0
for yz in [0.,0.5,1.,1.5]: # števila polov
    plt.plot(x,fun1(x,yz), color=barva[c], alpha = 0.95,  label=r'$y_0$={}'.format(yz))
    c=c+1
    
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1.2])
plt.legend(loc='best')

#################################################################################

F660=plt.figure(660)
#F60=plt.subplot(1, 1, 1 ) 
plt.title('Vožnja do semaforja za periodični robni pogoj',fontsize=16)
x=np.linspace(0,1,100)#-len(FTSIG)/2

c=0
for yz in [0.,0.5,1.,1.5]: # števila polov
    plt.plot(x,fun5(x,yz), color=barva[c], alpha = 0.95,  label=r'$y_0$={}'.format(yz))
    c=c+1
    
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1.2])
plt.legend(loc='best')


##################################################################################


F10=plt.figure(10)
#F60=plt.subplot(1, 1, 1 ) 
plt.title('Vožnja do semaforja za omejitev končne hitrosti',fontsize=16)

d=0
for yk in [0.5,1,1.5]: # števila polov
    c=0
    for yz in [0.5,1.,1.5]: # števila polov
        plt.plot(x,fun2(x,yz,yk), color=barva[c],ls=LS[d], alpha = 0.95, label=r'$y_0$={}'.format(yz)+r',  $y_T$={}'.format(yk))
        c=c+1
    d=d+1

plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1.2])
plt.legend(loc='best')


##################################################################################


F11=plt.figure(11)

d=1
for p in [1,2,3]: # števila polov
    F11=plt.subplot(2, 2, d ) 
    plt.title(r'Vožnja do semaforja za prosti robni pogoj pri potenci '+r'$p$={}'.format(p),fontsize=16)

    c=0
    for yz in [0,0.5,1.,1.5]: # števila polov
        plt.plot(x,fun3(x,yz,p), color=barva[c], alpha = 0.95, label=r'$y_0$={}'.format(yz) )
        c=c+1
    d=d+1
    
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y(x)$',fontsize=16)
    plt.xlim([0,1.3])
    plt.legend(loc='best')
    

F11=plt.subplot(2, 2, 4 ) 
plt.title(r'Vožnja do semaforja za prosti robni pogoj pri potenci $p=\infty$',fontsize=16)
c=0
for yz in [0,0.5,1.,1.5]: # števila polov
    plt.plot(x,fun33(x,yz),  color=barva[c] , alpha = 0.95, label=r'$y_0$={}'.format(yz)+r',  ')
    c=c+1
    
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1.3])
plt.legend(loc='best')


##################################################################################


F111=plt.figure(111)


d=1
for mu in [1,5,10,100]: # števila polov
    F111=plt.subplot(2, 2, d ) 
    plt.title(r'Vožnja do semaforja za prosti robni pogoj pri '+r'$\mu$={}'.format(mu),fontsize=16)

    c=0
    for yz in [0,0.5,1.,1.5]: # števila polov
        plt.plot(x,fun4(x,yz,mu), color=barva[c], alpha = 0.95, label=r'$y_0$={}'.format(yz) )
        c=c+1
    d=d+1
    
    plt.xlabel(r'$x$',fontsize=16)
    plt.ylabel(r'$y(x)$',fontsize=16)
    plt.xlim([0,1.3])
    plt.legend(loc='best')
    