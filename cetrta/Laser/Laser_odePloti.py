# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:37:54 2017

@author: jernej
"""

# zombie apocalypse modeling
import pylab 
import numpy as np
import matplotlib.pyplot  as plt
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.signal import argrelmax

#plt.ion()
#plt.rcParams['figure.figsize'] = 10, 8

# write the system dy/dt = f(y, t)
#def f(t,y):
def f(t,y,B,D,C,E,Q):
     F = y[0]
     A = y[1]
#     Ri = y[2]
#     # the model equations (see Munz et al. 2009)
#     f0 = P - B*Si*Zi - d*Si
#     f1 = B*Si*Zi + G*Ri - A*Si*Zi
#     f2 = d*Si + A*Si*Zi - G*Ri
     
     dFdt = -B*F -D*A*F
     dAdt = -C*A -E*A*F +Q     
     
#     return [f0, f1, f2]
     return [dFdt, dAdt]#, f2]

#def fBD(t,y):
def fBD(t,y,alfa,beta,gamma,delta):
     Z = y[0]
     L = y[1]
#     Ri = y[2]
#     # the model equations (see Munz et al. 2009)
#     f0 = P - B*Si*Zi - d*Si
#     f1 = B*Si*Zi + G*Ri - A*Si*Zi
#     f2 = d*Si + A*Si*Zi - G*Ri
     l=L*beta/alfa
     z=Z*delta/gamma
     p=np.sqrt(alfa/gamma)

     dZdt = p*z - p*z*l
     dLdt = l*z/p + l/p     
     
#     return [f0, f1, f2]
     return [dZdt, dLdt]#, f2]
     

#
#def fun(t, z, omega):
#    """
#    Right hand side of the differential equations
#      dx/dt = -omega * y
#      dy/dt = omega * x
#    """
#    x, y = z
#    f = [-omega*y, omega*x]
#    return f

# Create an `ode` instance to solve the system of differential
# equations defined by `fun`, and set the solver method to 'dop853'.
#solver = ode(fBD)
solver = ode(f)
solver.set_integrator('dopri5')
#solver.set_integrator('dop853')

# Give the value of omega to the solver. This is passed to
# `fun` when the solver calls it.
j=0
while j<1:
    
    KOEF=[[1,1,1,1,1],[0.1,0.1,2,2],[2,2,0.1,0.1],[1,2,2,1],[2,1,1,2]]
    
    koef=KOEF[j]
    B = koef[0]  # izsevani fotoni
    D = koef[1]  # nastali fotoni
    C = koef[2]  # termično relaksirani atomi
    E = koef[3]  # fotonsko relaksirani atomi  
    Q = koef[4]  # črpanje
    
    i=1    
#    D = np.empty(28)
    Q = np.empty(8)
    while i <9:
        
        solver.set_f_params(B,D,C,E,Q)
        
        Q0 = 1           # initial population
        q= 0.101*i
        Q[i-1]=q       
        F0 = a*A0            # initial zombie population
        y0 = [F0, A0]       # initial condition vector
                      
        t0 = 0.0
        t1 = 50
        N = 1000
        t = np.linspace(t0, t1, N)
        
        solver.set_initial_value(y0, t0)
        soln = np.empty((N, 2))
        soln[0] = y0
        
        # Repeatedly call the `integrate` method to advance the
        # solution to time t[k], and save the solution in sol[k].
        k = 1
        while solver.successful() and solver.t < t1:
            solver.integrate(t[k])
            soln[k] = solver.y
            k += 1
        
        Z = soln[:, 0]
        L = soln[:, 1]
        
        # plot results
        crta=['b','r','m','g','k','c','y']
        
#        if i==1 or i==6:
#            F1=plt.subplot(1, 2, 1 )
#            plt.plot(t, Z, '-',c=crta[i-1],label='Zajci a='+'{:.{}f}'.format(a, 2 ))
#            plt.plot(t, L, '--',c=crta[i-1],label='Lisice a='+'{:.{}f}'.format(a, 2 ))
#            plt.plot(t, OHRANITEV, ':',c=crta[i-1],label='Ohranitev a='+'{:.{}f}'.format(a, 2 ))
#            plt.xlabel('Št. dni')
#            plt.ylabel('Populacija')
#    #        plt.title('Populacijska dinamika: lisice - zajci  ('+r'$\beta/\alpha$='+str(beta/alfa)+ r', $\delta/\gamma$='+'{:.{}f}'.format(delta/gamma, 3 )+')')
#            plt.title('Populacijska dinamika: lisice - zajci  '+'('+r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+')')
#    
#            plt.legend(loc=1)
#        
#        F2=plt.subplot(1, 2, 2 )
#    #    plt.scatter(Z,L, c=crta[i-1],marker=",", label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta))
#        plt.scatter(Z,L, c=crta[i-1],marker="x",label='a='+'{:.{}f}'.format(a, 2 ))   
#        plt.xlabel('Zajci')
#        plt.ylabel('Lisice')
#    #    plt.title('Fazni diagram: lisice - zajci')
#        plt.title('Fazni diagram:  lisice - zajci  '+'('+r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+')')
#        plt.legend(loc=0)
        
        
#        MAX=argrelmax(Z)[0]
#        M=MAX*t1/N
#        d=(M[3]-M[2]+M[2]-M[1])/2
        
        i=i+1
    
#    MIN=np.amax(-D)
    
#    crta=['b','r','m','g','k','c','y']
    CRTA=['-','--','-.',':']
    Dash=[[10, 5, 20, 5],[2,2,10,2,2]]    
    F1=plt.subplot(1, 2, 1 )
    if j<4:
        plt.plot(Q, D, c='k',ls=CRTA[j],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN, 1 )+')')
    else:
        plt.plot(Q, D, c='k',ls='--',dashes=Dash[j-4],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN, 1 )+')')
    plt.xlabel('Moč črpanja')
    plt.ylabel('Obhodna doba', color='k')
    plt.title('Obhodne dobe v odvisnosti od začetnega stanja lisice/zajci' )
    plt.legend(loc=0)
    #        
    F2=plt.subplot(1, 2, 2 )   
    if j<4:
        plt.plot(A, O, c='b',ls=CRTA[j],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN_O, 1 )+')')
    else:
        plt.plot(A, O, c='b',ls='--',dashes=Dash[j-4],label=r'$\alpha$='+str(alfa)+r', $\beta$='+str(beta)+r', $\gamma$='+str(gamma)+r', $\delta$='+str(delta)+r', $(Min$='+'{:.{}f}'.format(-MIN_O, 1 )+')')
    plt.xlabel('L/Z')    
    plt.ylabel('Konstanta ohranitve')
    plt.title('Vrednost konstante ohranitve v odvisnosti od začetnega stanja lisice/zajci' )

    plt.legend(loc=0)
    

    j=j+1
    
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