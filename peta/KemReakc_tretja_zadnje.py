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
def f(t,y,a,b):

     SO8 = y[0]
     I = y[1]
     I2= y[2]
     SO3= y[3]
     
     dSO8dt = -a*SO8*I
     dIdt = -2*a*SO8*I + 2*b*SO3*I2
     dI2dt = a*SO8*I - b*SO3*I2
     dSO3dt = -2*b*SO3*I2
#     dH2dt = dB2dt
     
     return [dSO8dt, dIdt, dI2dt,dSO3dt]#, dBdt, dCdt]#, f2]

def fstac(t,y,a,b):
    
     SO8 = y[0]
     I = y[1]
     I2= y[2]
     SO3= y[3]
     
     dSO8dt = -a*SO8*I
     dIdt = -a*SO8*I**2+b*SO3*I2**2
     dI2dt = a*SO8*I**2-b*SO3*I2**2
     dSO3dt = -b*SO3*I2
#     return [f0, f1, f2]
     return [dSO8dt, dIdt, dI2dt,dSO3dt]#, f2]


i=0
while i<1:
    
    SO8 = [0,0,0,0]
    I = [0,0,0,0]
    I2= [0,0,0,0]
    SO3= [0,0,0,0]
    SO30= [0,0,0,0]
    r=[0,0,0,0]
        
    so8=1
#    SO30=[0.1,1,10]    
#    SO30=0.5
#    KOEF=[[so8,1,0,SO30],[so8,0,1,SO30],[so8,1,1,SO30]]
#    KOEF=[[100,1,10000],[50.5,50.5,10000],[1,100,10000]]

    I0 = KOEF[i][1]           # initial population
    I20 = KOEF[i][2]           # initial population#    B20 = 1.
    so3 = KOEF[i][3] 
    y0 = KOEF[i] #[H20, B20, HB0]#, C0]       # initial condition vector
    
    j=0
    while j<4:
        
#        so8=1
#        SO30=[0.1,1,10]    
        SO30[j]=0.5*(j+1)
        KOEF=[[so8,1,0,SO30[j]],[so8,0,1,SO30],[so8,1,1,SO30]]
        y0 = KOEF[i]
    
        koef=[[1,1],[10,1],[100,1],[1000,1]]
        b = koef[3][0]
        a = koef[3][1]  # množenje zajcev
    # pohrustani zajci (per day)
        r=b/a
        
        t0 = 0.
        t1 = 5
        N = 1000
        t = np.linspace(t0, t1, N)
    
        solver = ode(f)
#        solver = ode(fstac)
    #    solver.set_integrator('dopri5')
        solver.set_integrator('dop853')       
        solver.set_f_params(a,b)    
        solver.set_initial_value(y0, t0)
        soln = np.empty((N, 4))
        soln[0][0] = KOEF[i][0]
        soln[0][1] = KOEF[i][1]    
        soln[0][2] = KOEF[i][2]  
        soln[0][3] = KOEF[i][3]  
    
        # Repeatedly call the `integrate` method to advance the
        # solution to time t[k], and save the solution in sol[k].
        k = 1
        while solver.successful() and solver.t < t1:
            solver.integrate(t[k])
            soln[k] = solver.y
            k += 1
             
    #    h2= soln[:, 0] 
    #    b2= soln[:, 1] 
    #    hb = soln[:, 2] 
    #    hb = soln[:, 3] 
        
        SO8[j] = soln[:, 0] 
        I[j] = soln[:, 1] 
        I2[j] = soln[:, 2] 
        SO3[j] = soln[:, 3] 
        
    
        print(j)
        j=j+1
    
    # plot results
    crta=['k','c','m','y','b','r']
    
#    if i==0:
    F1=plt.subplot(2, 1, 1 )
#    if i==1:
#        F2=plt.subplot(3, 1, 2 )
#    if i==2:
#        F3=plt.subplot(3, 1, 3 )
#    if i==3:
#        F2=plt.subplot(4, 1, 4 )
    j=0
    while j<3:
    #    if i==1 or i==6:
        plt.plot(t, SO8[j]-so8, ':',c=crta[j],label=r'$[S_2O^{-2}_8]-[S_2O^{-2}_8]_0$ , '+r'$[S_2O^{-2}_3]_0=$'+str(SO30[j]))# / q \cdot A_0=$'+'{:.{}f}'.format(r, 1 ))
        plt.plot(t, I[j], '--',c=crta[j],label=r'$[I^-]$ , '+r'$[S_2O^{-2}_3]_0=$'+str(SO30[j]))#  '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    #    plt.plot(t, A2[j], '-.',c=crta[j],label=r'$A^*_ {eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    #    plt.plot(t, A2s[j], ':',c=crta[j],label=r'$A^*_ {stac}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    #    plt.plot(t, HB[j], '-',c=crta[j],label=r'$[HBr_2]$ , '+'r='+str(r[j]))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
        plt.plot(t, I2[j], '-.',c=crta[j],label=r'$[I_2]$ , '+r'$[S_2O^{-2}_3]_0=$'+str(SO30[j]))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
        plt.plot(t, SO3[j], '-',c=crta[j],label=r'$[S_3O^{-2}_3]$ , '+r'$[S_2O^{-2}_3]_0=$'+str(SO30[j]))#+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))   
        j=j+1
#    plt.ylim([10**(-7),10**(7)])
    plt.xlabel('čas')
    plt.ylabel('koncentracija')#+r'$A,B$')
    plt.title('Dinamika koncentracij reakcije kemijske ure za r='+str(r)+' in '+r'$[1,1,0,[S_2O^{-2}_3]_0]$')#+str(KOEF[i]))#+r'$A$'+' in '+r'$B\ \ (oz.\ C)$')
    plt.legend(loc=1)
#    plt.yscale('log')
    
    i=i+1
    
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
