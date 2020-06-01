# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:37:54 2017

@author: jernej
"""

# zombie apocalypse modeling
#import pylab 
import numpy as np
import matplotlib.pyplot  as plt
#from scipy.integrate import odeint
from scipy.integrate import ode
#from scipy.signal import argrelmax

#plt.ion()
#plt.rcParams['figure.figsize'] = 10, 8

# write the system dy/dt = f(y, t)
#def f(t,y):
def f(t,y,k,m):

     H2 = y[0]
     B2 = y[1]
     HB= y[2]
     
     dH2dt = -2* (k* H2*np.sqrt(B2)/(m+(HB/B2)))
     dB2dt = -2* (k* H2*np.sqrt(B2)/(m+(HB/B2)))
     dHBdt = k*H2*np.sqrt(B2)/(m+(HB/B2))
#     dB2dt = -2* dHBdt
#     dH2dt = dB2dt
     
     return [dH2dt, dB2dt, dHBdt]#, dBdt, dCdt]#, f2]

#def fstac(t,y,p,q,r):    
#     A1 = y[0]
#     
#     dA1dt = -r* p* A1**2. /(q* A1 + r)
#
##     return [f0, f1, f2]
#     return [dA1dt]#, f2]
     

H2 = [0,0,0]
B2 = [0,0,0]
HB = [0,0,0]
#Bs  = [0,0,0]
r=[0,0,0]

j=0
while j<3:
    
    KOEF=[[100,1,0],[50.5,50.5,0],[1,100,0]]
#    KOEF=[[100,1,10000],[50.5,50.5,10000],[1,100,10000]]

#    H20 = KOEF[j]           # initial population
#    B20 = 1.
#    HB0 = 0.
#    HB0 = 10000.
    r[j]=KOEF[j][0]/KOEF[j][1]
    
    y0 = KOEF[j] #[H20, B20, HB0]#, C0]       # initial condition vector
    
    k = 1.  # množenje zajcev
    m = 2.5  # pohrustani zajci (per day)
    
    t0 = 0.
    t1 = 2.
    N = 1000
    t = np.linspace(t0, t1, N)

    solver = ode(f)
#    solver.set_integrator('dopri5')
    solver.set_integrator('dop853')       
    solver.set_f_params(k,m)    
    solver.set_initial_value(y0, t0)
    soln = np.empty((N, 3))
    soln[0] = KOEF[j][0]
    soln[1] = KOEF[j][1]    
    soln[2] = KOEF[j][2]    

    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < t1:
        solver.integrate(t[k])
        soln[k] = solver.y
        k += 1
         
    h2= soln[:, 0] 
    b2= soln[:, 1] 
    hb = soln[:, 2] 
    
    H2[j] = h2
    B2[j] = b2
    HB[j] = hb
    

    print(j)
    j=j+1

# plot results
crta=['k','c','m','y','b','r']

#F1=plt.subplot(2, 1, 1 )
F2=plt.subplot(2, 1, 2 )
j=0
while j<3:
#    if i==1 or i==6:
    plt.plot(t, H2[j], '-.',c=crta[j],label=r'$[H_2]$ , '+'r='+str(r[j]))# / q \cdot A_0=$'+'{:.{}f}'.format(r, 1 ))
    plt.plot(t, B2[j], '--',c=crta[j],label=r'$[Br_2]$ , '+'r='+str(r[j]))#  '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, A2[j], '-.',c=crta[j],label=r'$A^*_ {eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, A2s[j], ':',c=crta[j],label=r'$A^*_ {stac}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, HB[j], '-',c=crta[j],label=r'$[HBr_2]$ , '+'r='+str(r[j]))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
    plt.plot(t, HB[j]-KOEF[j][2], '-',c=crta[j],label=r'$[HBr]-[HBr]_0$ , '+'r='+str(r[j]))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
    plt.plot(t, H2[j]/B2[j], ':',c=crta[j],label=r'$[H_2]/[Br_2]$ , '+'r='+str(r[j]))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
    j=j+1
plt.ylim([10**(-7),10**(7)])
plt.xlabel('čas')
plt.ylabel('koncentracija')#+r'$A,B$')
plt.title('Dinamika koncentracij večstopenjske kemijske reakcije za m=2.5, k=1 in '+r'$[HBr]_0=$'+str(KOEF[0][2]))#+r'$A$'+' in '+r'$B\ \ (oz.\ C)$')
plt.legend(loc=1)
plt.yscale('log')


##HB0=100
##H201 = 0.001     
##B201 = 1
##H202 = 1  # initial population
##B202 = 0.001
##y01 = [H201, B201, HB0]#, C0]       # initial condition vector
##y02 = [H202, B202, HB0]#, C0]       # initial condition vector
#HB0=0
##HB0=10
#KOEF=[[1,0.001,HB0],[50.5,50.5,HB0],[0.001,1,HB0]]
##KOEF=[[0.001,1,HB0],[50.5,50.5,HB0],[1,0.001,HB0]]
#solver = ode(f)
##    solver.set_integrator('dopri5')
#solver.set_integrator('dop853')       
#solver.set_f_params(k,m)    
#solver.set_initial_value(KOEF[2], t0)
#soln = np.empty((N, 3))
#soln[0] = KOEF[2][0]
#soln[1] = KOEF[2][1]
#soln[2] = KOEF[2][2]
#
## Repeatedly call the `integrate` method to advance the
## solution to time t[k], and save the solution in sol[k].
#k = 1
#while solver.successful() and solver.t < t1:
#    solver.integrate(t[k])
#    soln[k] = solver.y
#    k += 1
#     
#h2= soln[:, 0] 
#
#solver = ode(f)
##    solver.set_integrator('dopri5')
#solver.set_integrator('dop853')       
#solver.set_f_params(k,m)    
#solver.set_initial_value(KOEF[0], t0)
#soln = np.empty((N, 3))
#soln[0] = KOEF[0][0]
#soln[1] = KOEF[0][1]     
#soln[2] = KOEF[0][2]    
#
## Repeatedly call the `integrate` method to advance the
## solution to time t[k], and save the solution in sol[k].
#
#k = 1
#while solver.successful() and solver.t < t1:
#    solver.integrate(t[k])
#    soln[k] = solver.y
#    k += 1
#     
#b2= soln[:, 1] 
#
#    
#th=np.empty(N)
#tb=np.empty(N)
#th[0]=0
#C=-(8/np.sqrt(1 + KOEF[2][0])) + 4 *m* np.arctan(np.sqrt(1 + KOEF[2][0]))
#tb[0]=0
#D=-4*(-2 + m)*np.arctan(np.sqrt(KOEF[0][1]))
#i=1
#while i<N:
##    th[i] = -2.*m*np.log(h2[i]/KOEF[2][0])+4*(h2[i]-KOEF[2][0])
##    tb[i] = -4.*(m-2.)*(np.sqrt(b2[i])-np.sqrt(KOEF[0][1]))+8*KOEF[0][1] *(1./np.sqrt(b2[i])-1/np.sqrt(KOEF[0][1] ))
#    th[i] = -(8/np.sqrt(1 + h2[i])) + 4 *m* np.arctan(np.sqrt(1 + h2[i]))-C
#    tb[i] = -4 *(-2 + m)* np.arctan(np.sqrt(b2[i]))-D
#    i=i+1
#
## plot results
#F3=plt.subplot(2, 1, 2 )
##plt.plot(t, np.abs(t-th), '--',c=crta[j+1],label=r'$t-t[H_2]$ , '+'r='+str(KOEF[2][0]/KOEF[2][1]))#+str(1/1000))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
##plt.plot(t, np.abs(t-tb), ':',c=crta[j+1],label=r'$t-t[Br_2]$ , '+'r='+str(KOEF[0][0]/KOEF[0][1] ))#+str(1000))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
#plt.plot(t, th, '--',c=crta[j],label=r'$t[H_2]$ , '+'r='+str(KOEF[2][0]/KOEF[2][1]))#+str(1/1000))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
#plt.plot(t, tb, ':',c=crta[j],label=r'$t[Br_2]$ , '+'r='+str(KOEF[0][0]/KOEF[0][1] ))#+str(1000))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
#plt.plot(t, np.abs(th-tb), '-',c=crta[j],label=r'$t[H_2]-t[Br_2]$')# , '+'r='+str(H202/B202))#+str(1000))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
#plt.xlabel('čas')
#plt.ylabel(r'$\Delta t$')
#plt.title('Primerjava časa s časom računanim iz koncentracij večstopenjske kemijske reakcije za m=2.5, k=1 in '+r'$[HBr]_0=$'+str(KOEF[0][2]))#+r'$A$'+' in '+r'$B\ \ (oz.\ C)$')
#plt.legend(loc=1)
##plt.yscale('log')
