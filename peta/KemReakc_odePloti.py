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
def f(t,y,p,q,r):
     A1 = y[0]
     A2 = y[1]
    
     dA1dt = - p* A1**2. + q * A1 * A2
     dA2dt = p* A1**2. - q * A1 * A2 - r * A2
     dBdt =  + r * A2
#     dCdt = r * A2

     return [dA1dt, dA2dt, dBdt]#, dBdt, dCdt]#, f2]

def fstac(t,y,p,q,r):    
     A1 = y[0]
     
     dA1dt = -r* p* A1**2. /(q* A1 + r)

#     return [f0, f1, f2]
     return [dA1dt]#, f2]
     

A1 = [0,0,0]
A1s = [0,0,0]

A2 = [0,0,0]
A2s = [0,0,0]

B = [0,0,0]
Bs  = [0,0,0]

j=0
while j<3:
    
    KOEF=[10000,1000,100]
    
    p = 1.  # množenje zajcev
    q = 1000.  # pohrustani zajci (per day)
    r = KOEF[j]  # smrt  (per day)
    
#    i=1
#    D = np.empty(28)
#    A = np.empty(28)
#    O = np.empty(28)
#    while i <29:
    
    A10 = 1.           # initial population
    A20 = 0.
    B0 = 0.
#    C0 = 0
    y0 = [A10, A20, B0]#, C0]       # initial condition vector
                  
    t0 = 0.
    t1 = 100.
    N = 1000
    t = np.linspace(t0, t1, N)

    solver = ode(f)
#    solver.set_integrator('dopri5')
    solver.set_integrator('dop853')       
    solver.set_f_params(p,q,r)    
    solver.set_initial_value(y0, t0)
    soln = np.empty((N, 3))
    soln[0] = A10    
    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < t1:
        solver.integrate(t[k])
        soln[k] = solver.y
        k += 1

    A1[j] = soln[:, 0] 
    A2[j] = soln[:, 1]
    B[j] = soln[:, 2]
     
    
#    B = soln[:, 2]
#    C = soln[:, 3]
    
    solver = ode(fstac)
#    solver.set_integrator('dopri5')
    solver.set_integrator('dop853')
    solver.set_f_params(p,q,r)    
    solver.set_initial_value(y0[0], t0)
    solns = np.empty((N, 1))
    solns[0] = A10    
    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < t1:
        solver.integrate(t[k])
        solns[k] = solver.y
        k += 1
    A1s[j] = solns[:, 0]
    a2s = p* (solns[:, 0])**2. /(q* (solns[:, 0]) + r)
    A2s[j]=a2s
    bs = np.empty(1000)
    bs[0]=0
    k=1
    while k<N:
        bs[k] = bs[k-1] + r * a2s[k-1] * (t1-t0)/N
        k=k+1
#    Cs = r * A2s
    Bs[j]=bs
    
    j=j+1


# plot results
crta=['k','c','m','y','b','r']

F1=plt.subplot(3, 1, 1 )
j=0
while j<3:
#    if i==1 or i==6:
    plt.plot(t, A1[j], '-',c=crta[j],label=r'$A_{eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    plt.plot(t, A1s[j], '--',c=crta[j],label=r'$A_ {stac}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, A2[j], '-.',c=crta[j],label=r'$A^*_ {eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, A2s[j], ':',c=crta[j],label=r'$A^*_ {stac}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    plt.plot(t, B[j], ':',c=crta[j],label=r'$B_ {eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, Bs[j], ':',c=crta[j+3],label=r'$B_ {stac}$'+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    j=j+1
plt.xlabel('čas')
plt.ylabel('populacija  '+r'$A,B$')
plt.title('Dinamika binarne kemijske reakcije za koncentracije '+r'$A$'+' in '+r'$B\ \ (oz.\ C)$')
plt.legend(loc=1)
plt.yscale('log')

F2=plt.subplot(3, 1, 2 )
j=0
while j<3:
#    if i==1 or i==6:
#    plt.plot(t, A1[j], '-',c=crta[j],label=r'$A_{eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, A1s[j], '--',c=crta[j],label=r'$A_ {stac}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    plt.plot(t, A2[j], '-.',c=crta[j],label=r'$A^*_ {eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    plt.plot(t, A2s[j], ':',c=crta[j],label=r'$A^*_ {stac}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, B[j], '-',c=crta[j+3],label=r'$B_ {eks}$   '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t, Bs[j], ':',c=crta[j+3],label=r'$B_ {stac}$'+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    j=j+1
plt.xlabel('čas')
plt.ylabel('populacija  '+r'$A^*$')
plt.title('Dinamika binarne kemijske reakcije za koncentracijo '+r'$A^*$')
plt.legend(loc=1)
plt.yscale('log')

F3=plt.subplot(3, 1, 3 )
j=0
while j<3:    
    plt.plot(t,A1[j]-A1s[j],'--', c=crta[j],label=r'$\Delta A$    '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
    plt.plot(t,A2[j]-A2s[j],':', c=crta[j],label=r'$\Delta A^*$    '+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(10**(-j+4)/(q*A10), 1 ))
#    plt.plot(t,B[j]-Bs[j],':', c=crta[j])#,label=r'$A^*$'+r'$r / q \cdot A_0=$'+'{:.{}f}'.format(r/(q*A0), 1 ))
    j=j+1
plt.xlabel('čas')
plt.ylabel(r'$A_{eks}-A_ {stac}$')
plt.title('Odstopanja med rešitvami eksaktne in stacionarne obravnave')
plt.legend(loc=1)
plt.yscale('log')
    
#        
#        MAX=argrelmax(Z)[0]
#        M=MAX*t1/N
#        d=(M[3]-M[2]+M[2]-M[1])/2
#        
#        D[i-1]=d
#        O[i-1]=ohranitev
#    i=i+1
    
#    MIN=np.amax(-D)
#    MIN_O=np.amax(-O)
##    crta=['b','r','m','g','k','c','y']
#    CRTA=['-','--','-.',':']
#    Dash=[[10, 5, 20, 5],[2,2,10,2,2]]    
#    
#    F1=plt.subplot(1, 2, 1 )
#    if j<4:
#        plt.plot(A, D, c='k',ls=CRTA[j],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN, 1 )+')')
#    else:
#        plt.plot(A, D, c='k',ls='--',dashes=Dash[j-4],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN, 1 )+')')
#    plt.xlabel('L/Z')
#    plt.ylabel('Obhodna doba', color='k')
#    plt.title('Obhodne dobe v odvisnosti od začetnega stanja lisice/zajci' )
#    plt.legend(loc=0)
#    #        
#    F2=plt.subplot(1, 2, 2 )   
#    if j<4:
#        plt.plot(A, O, c='b',ls=CRTA[j],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN_O, 1 )+')')
#    else:
#        plt.plot(A, O, c='b',ls='--',dashes=Dash[j-4],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN_O, 1 )+')')
#    plt.xlabel('L/Z')    
#    plt.ylabel('Konstanta ohranitve')
#    plt.title('Vrednost konstante ohranitve v odvisnosti od začetnega stanja lisice/zajci' )
#
#    plt.legend(loc=0)
    

    
    
#___BREZDIMENZIJSKO___
#
#l=L*beta/alfa
#z=Z*delta/gamma
#tau = t*np.sqrt(alfa*gamma) # time grid brezdimenzijsko
#
#BDsoln = odeint(fBD, y0, tau)
#z = BDsoln[:, 0]
#l = BDsoln[:, 1]
##
# plot results
#FIG1BD=plt.subplot(1, 2, 1 )
#plt.plot(tau, z, label='Zajci')
#plt.plot(tau, l, label='Lisice')
#plt.xlabel('Št. dni')
#plt.ylabel('Populacija')
#plt.title('Populacijska dinamika: lisice-zajci  ('+r'$\beta/\alpha$='+str(beta/alfa)+ r', $\delta/\gamma$='+'{:.{}f}'.format(delta/gamma, 3 )+')')
#plt.legend(loc=0)
##
#FIG2BD=plt.subplot(1, 2, 2 )
#plt.scatter(Z,L,'k',label=r'$\beta/\alpha$='+str(beta/alfa)+ r', $\delta/\gamma$='+'{:.{}f}'.format(delta/gamma, 3 ))
#plt.xlabel('Zajci '+ r'$\cdot\ \delta/\gamma$')
#plt.ylabel('Lisice '+ r'$\cdot\ \beta/\alpha$')
#plt.title('Fazni diagram: lisice-zajci')
#plt.legend(loc=0)

## change the initial conditions
#R0 = 0.01*S0   # 1% of initial pop is dead
#y0 = [S0, Z0, R0]
#
## solve the DEs
#soln = odeint(f, y0, t)
#S = soln[:, 0]
#Z = soln[:, 1]
#R = soln[:, 2]
#
#plt.figure()
#plt.plot(t, S, label='Living')
#plt.plot(t, Z, label='Zombies')
#plt.xlabel('Days from outbreak')
#plt.ylabel('Population')
#plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; No New Births.')
#plt.legend(loc=0)
#
## change the initial conditions
#R0 = 0.01*S0   # 1% of initial pop is dead
#P  = 10        # 10 new births daily
#y0 = [S0, Z0, R0]
#
## solve the DEs
#soln = odeint(f, y0, t)
#S = soln[:, 0]
#Z = soln[:, 1]
#R = soln[:, 2]
#
#plt.figure()
#plt.plot(t, S, label='Living')
#plt.plot(t, Z, label='Zombies')
#plt.xlabel('Days from outbreak')
#plt.ylabel('Population')
#plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; 10 Daily Births')
#plt.legend(loc=0)