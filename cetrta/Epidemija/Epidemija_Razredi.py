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
def f(t,y,alfa,beta1,beta2,beta3):
     D = y[0]
     B1 = y[1]
     B2 = y[2]
     B3 = y[3]
     I = y[4]
#     Ri = y[2]
#     # the model equations (see Munz et al. 2009)
#     f0 = P - B*Si*Zi - d*Si
#     f1 = B*Si*Zi + G*Ri - A*Si*Zi
#     f2 = d*Si + A*Si*Zi - G*Ri
     
     dDdt = -alfa*D*(B1+B2+B3)
     dB1dt = +alfa*D*(B1+B2+B3) -beta1*B1     
     dB2dt = +beta1*B1 -beta2*B2
     dB3dt = +beta2*B2 -beta3*B3
     dIdt = +beta3*B3
#     return [f0, f1, f2]
     return [dDdt, dB1dt, dB2dt,dB3dt,dIdt]#, f2]

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
#solver.set_integrator('dopri5')
solver.set_integrator('dop853')

# Give the value of omega to the solver. This is passed to
# `fun` when the solver calls it.

#KOEF=[[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,2],[1,1,1,1,0],[0.9,0.9,0.2,0.2,1],[1,2,1,2,1]]
#    
#koef=KOEF[i]
#B = koef[0]  # izsevani fotoni
#D = koef[1]  # nastali fotoni
#C = koef[2]  # termično relaksirani atomi
#E = koef[3]  # fotonsko relaksirani atomi  
#Q = koef[4]  # črpanje

j=0
i=0
#D = np.empty(28)
#q = np.empty(35)
#BA = np.empty(100)
#Bmax = np.empty(100)
#Tmax = np.empty(100)
#Bvsi = np.empty(100)
while i <4:
    
#    KOEF=[[1,1,1,1,1],[1,1,1,1,2],[1,1,0,1,2],[1,1,1,1,0],[1,3,1,1,1],[1,0.9,0.01,1,1]] 
#    koef=KOEF[5]
##    koef=KOEF[i]
#    B = koef[0]  # izsevani fotoni
#    D = koef[1]  # nastali fotoni
#    C = koef[2]  # termično relaksirani atomi
#    E = koef[3]  # fotonsko relaksirani atomi  
#    Q = koef[4]  # črpanje  
#    
#    Q= Q*i*0.1
#    
#    solver.set_f_params(B,D,C,E,Q)
    
    beta1=3
    beta2=1
    beta3=0.1
    alfa=2**(i)
    solver.set_f_params(alfa,beta1,beta2,beta3)

    
    D0 = 1
    a= 0.01#*i#+0.001  # initial population
    B10 =a*D0
    B20 =a*D0
    B30 =a*D0
    b= 10.8#*i#+0.001  # initial population
    I0 =b*D0
    y0 = [D0, B10, B20, B30, I0]            # initial zombie population
#    y0 = [0.01, 0, ]            # initial zombie population

#    q= 0.101*i
#    Q[i-1]=q       
                  
    t0 = 0.0
    t1 = 200
    N = 1000
    t = np.linspace(t0, t1, N)
    
    solver.set_initial_value(y0, t0)
    soln = np.empty((N, 5))
    soln[0] = y0
    
    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < t1:
        solver.integrate(t[k])
        soln[k] = solver.y
        k += 1
    
    D = soln[:, 0]
    B1 = soln[:, 1]
    B2 = soln[:, 2]
    B3 = soln[:, 3]
    I = soln[:, 4]
#    bolni=np.cumsum(B)
    # plot results
    crta=['k','b','r','m','g','c','y']
    
#    if i==0 or i==1 or i==5:
#        F1=plt.subplot(1, 2, 1 )
#       plt.plot(t, F, '-',c=crta[i-1],label='izsevani fotoni (a='+'{:.{}f}'.format(a, 2 )+')')
#        plt.plot(t, A, '--',c=crta[i-1],label='vzbujeni atomi (a='+'{:.{}f}'.format(a, 2 )+')')
#        plt.xlabel('čas')
#        plt.ylabel('N')
##        plt.title('Časovna odvisnost populacije: fotoni - atomi '+'(B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
#
#        plt.legend(loc=0)
#    if alfa/beta==0.5 or alfa/beta==1.0 or alfa/beta==1.5 or alfa/beta==2.0 or alfa/beta==2.5 or alfa/beta== 3.0 or alfa/beta== 3.5:
#    if alfa/beta==0.5 or alfa/beta==1.5 or  alfa/beta==2.5 or alfa/beta== 3.5:
    
    F1=plt.subplot(1, 1, 1 )
#    plt.plot(t, F, '-',c=crta[i],label='('+str(B)+','+str(D)+','+str(C)+','+str(E)+','+str(Q)+')')
    plt.plot(t, D+B1+B2+B3+I, 'k:')#,label=r'$\alpha/ \beta$='+'{:.{}f}'.format(alfa/beta, 1 ))
    plt.plot(t, D, '-',c=crta[i],label='('+str(alfa)+','+str(beta1)+','+str(beta2)+','+str(beta3)+')')
    plt.plot(t, B1, '--',c=crta[i])#),label='vzbujeni atomi (B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
    plt.plot(t, B2, '-.',c=crta[i])#),label='vzbujeni atomi (B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
    plt.plot(t, B3, '--',c=crta[i],dashes=[2,2,10,2])#),label='vzbujeni atomi (B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
#    plt.plot(t, B1+B2+B3, '--',c=crta[i],dashes=[10,2,10,2])#),label='vzbujeni atomi (B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
    plt.plot(t, I, ':',c=crta[i])#),label='vzbujeni atomi (B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
    plt.ylim([0,D0+B10+B20+B30+I0])
    plt.xlabel('čas')
    plt.ylabel('velikost populacije')
#    plt.title('Časovna odvisnost populacij: fotoni - atomi   '+'(B,D,C,E,Q)')   
    plt.title('Časovna odvisnost razredov populacij: dovzetni - bolni - imuni  '+r'($\alpha , \beta_1 , \beta_2 , \beta_3$)')#+r'($\alpha\ ,\beta)$')       
    plt.legend(loc=1)
        
#        F2=plt.subplot(1, 2, 2 )
#    #    plt.scatter(A,F, c=crta[i-1],marker="x",label='a='+'{:.{}f}'.format(a, 2 ))   
#    #    plt.scatter(A,F, c=crta[i],marker="x",label='('+str(B)+','+str(D)+','+str(C)+','+str(E)+','+str(Q)+')')
#    #    plt.plot(t,np.add.accumulate(B), c=crta[i],label=r'$alpha/ \beta$='+'{:.{}f}'.format(alfa/beta, 1 ))
#        plt.plot(t,bolni/(D0+B0+I0), c=crta[j],label=r'$\alpha/ \beta$='+'{:.{}f}'.format(alfa/beta, 1 ))
#        plt.xlabel('čas')
#        plt.ylabel('število vseh zbolelih')
#    #    plt.ylim([0,3])    
#    #    plt.title('Fazni diagram: lisice - zajci')
#    #    plt.title('Fazni diagram:  fotoni- atomi '+'(B='+str(B)+',D='+str(D)+',C='+str(C)+',E='+str(E)+',Q='+str(Q)+')')
#    #    plt.title('Fazni diagram:  fotoni- atomi   '+'(B,D,C,E,Q)')   
#        plt.title('Časovna odvisnost populacije zbolelih' ) # '+'(1,1,1,1,Q)')           
#        plt.legend(loc=1)
    
#    j=j+1
    
#    BA[i]=alfa/beta
#    m = max(B)
#    T=[i for i, j in enumerate(B) if j == m]
#    Tmax[i]=T[0]*t1/N
#    Bmax[i]=m
#    Bvsi[i]=I[N-2]-I0+B[N-2]

#    MAX=argrelmax(Z)[0]
#    M=MAX*t1/N
#    d=M[3]-M[2]#+M[2]-M[1])/2
#    
#    q[i]=Q       
#    Fmax[i]=F[N-2]
#    Amax[i]=A[N-2]
    i=i+1
#    crta=['b','r','m','g','k','c','y']
#CRTA=['k-','k--','k-.','r:','r--','g--','b--.','r-']
#    
#F3=plt.subplot(1, 1, 1 )
#plt.plot(BA, Bvsi, 'k-')#,label='')
##plt.plot(BA, Bmax, 'k-')#,label='')
##plt.plot(q, Amax, 'k--',label='vzbujeni atomi')
#plt.xlabel(r'$\alpha/\beta$')
#plt.ylabel('vsi oboleli')
#plt.legend(loc=0)
#
#F4,ax1= plt.subplots()
#ax1.plot(BA, Tmax, 'k-')#,label='')
#ax1.set_xlabel(r'$\alpha/ \beta$')
#ax1.set_ylabel('čas maksimuma', color='k')
#ax1.tick_params('y', colors='k')
#plt.legend(loc=0)
#
#ax2 = ax1.twinx()
#ax2.plot(BA, Bmax, 'b-')#,label='')
#ax2.set_ylabel('vrednost maksimuma', color='b')
#ax2.tick_params('y', colors='b')
#plt.legend(loc=0)
#
#plt.title('Odvisnost lastnosti maksimuma populacije bolnih od razmerja  '+r'$\alpha/ \beta$')


#    plt.yscale('log’)
#    
#    j=j+1
#    
    
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