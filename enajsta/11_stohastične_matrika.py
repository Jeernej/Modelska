# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
#import scipy as sc
import matplotlib.pyplot  as plt
from scipy.sparse import diags
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
     

N0=250
N0=25
#N0=7

beta=1.
beta_R=4*beta ##rojstva
beta_S=5*beta ##smrti

N_T=np.arange(0,N0+1)  # vektor populacij od 0 do N0

dt=1/(N0*(beta_S+beta_R)) # časovni korak
#dt=0.0001

#### zgradi matriko ############################################

DIAG=np.empty(0)
ZG_DIAG=np.empty(0)
SP_DIAG=np.empty(0)
#
for n in range(0,N0+1):
        DIAG=np.append(DIAG, 1-(beta_R+beta_S)*N_T[n]*dt) ## glavna diagonala
for n in range(0,N0):
        SP_DIAG=np.append(SP_DIAG, beta_R * N_T[n] * dt) ## rojstva
for n in range(1,N0+1):
        ZG_DIAG=np.append(ZG_DIAG, beta_S * N_T[n] * dt) ## smrti

diagonals = [ZG_DIAG, DIAG, SP_DIAG]
M_t=diags(diagonals, [+1,0,-1]).toarray()


#for n in range(0,N0+1):
#        DIAG=np.append(DIAG, 1-(beta_R*N_T[n]+beta_S*N_T[n])*dt)       
#for n in range(1,N0+1):
#        ZG_DIAG=np.append(ZG_DIAG, beta * N_T[n] * dt) ## smrti
#
#diagonals = [ZG_DIAG, DIAG]
#M_t=diags(diagonals, [+1,0]).toarray()




##### zgradi začetno porazdelitev verjetnosti za velikosti populacije     
    
Pt=[np.zeros(N0+1)]
Pt[0][N0]=1

####################################################    

PT=[Pt[0]]
#PT=[]
#PT=np.empty(0)
T=np.empty(0)
avg=np.empty(0)
avg2=np.empty(0)
var=np.empty(0)

t_smrti=0

F50=plt.figure(50)
plt.plot(N_T,Pt[0],color='b',ls=':',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))

f=0
while Pt[0][0]<0.99:

#    Hkoc, xkoc, binEdge_koc, Lkoc=Hdata(Pt[0]*N0)
#    avg=np.append(avg,np.sum(Hkoc* np.diff(binEdge_koc)* xkoc ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
   
#    Hkoc, xkoc, binEdge_koc, Lkoc=Hdata((Pt[0]*N0)**2)
#    avg2=np.append(avg2,np.sum(Hkoc* np.diff(binEdge_koc)* xkoc ) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#    avg2=np.append(avg2, (Pt[0]*N0) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#    avg=np.append(avg, np.mean(Pt[0]*N0) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#    var=np.append(var, np.var(Pt[0*N_T/N0) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#    if (N0-20*f)/sum(Pt[0]*N_T)>1:
#        avg=np.append(avg, sum(Pt[0]*N_T) )#np.diff vrne razliko med vrednostmi sosednjih elementov
#        avg2=np.append(avg2, sum(Pt[0]*N_T**2 ))#np.diff vrne razliko med vrednostmi sosednjih elementov
#  
    
    Pt[0]=M_t.dot(Pt[0])
    Pt[0]=Pt[0]/sum(Pt[0])

#    Pt[0]=A.dot(Pt[0])
    t_smrti=t_smrti+dt

    if (N0-4*f)/sum(Pt[0]*N_T)>1:
    
        plt.plot(N_T,Pt[0],color='b',alpha = 0.95-f*0.1)#+'{:.{}f}'.format(SHfi, 3 ))

        PT=np.concatenate((PT, Pt), axis=0)

        avg=np.append(avg, sum(Pt[0]*N_T) )#np.diff vrne razliko med vrednostmi sosednjih elementov
        avg2=np.append(avg2, sum(Pt[0]*N_T**2 ))#np.diff vrne razliko med vrednostmi sosednjih elementov
 
    
#    if f>dt*100000:
        T=np.append(T, t_smrti)
        f=f+1
        
#    f=f+1
    
#var=np.sqrt(avg**2-avg2)    
var=(-avg**2+avg2)   
 
plt.plot(N_T,Pt[0],color='b',alpha = 0.95-f*0.1 , label=r'$N_0=$'+str(N0) +r',  $t_{izumrtja}=$'+'{:.{}f}'.format(t_smrti, 2)+', $dt=$'+'{:.{}f}'.format(dt, 4))#+'{:.{}f}'.format(SHfi, 3 ))

plt.title(r'Umiranje populacije')# računano z $dt=$'+'{:.{}f}'.format(dt, 4))#+r';  ($t_{izumrtja}=$'+'{:.{}f}'.format(t_smrti, 2)+')',fontsize=16)
plt.ylabel(' p(t) ')   
plt.xlabel(' N ')
plt.xlim([-1,250+1])
plt.ylim([0,1])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)
        
############################# plotanje 2D   
        
F60=plt.figure(60)
plt.title(r'Momenti porazdelitev')# za $dt=$'+'{:.{}f}'.format(dt, 4),fontsize=16)
#F50=plt.subplot(1, 2, 1 ) 
plt.plot(T,avg,color='k',ls='-',alpha = 0.95,label=r'$\bar{\mu} (t)$'+ r', $N_0=$'+str(N0) )#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,var,color='r',ls='-',alpha = 0.95,label=r'$\sigma (t)$'+r', $N_0=$'+str(N0))#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,N0*e**((beta_R-beta_S)*T),color='k',ls=':',alpha = 0.95,label=r'$\bar{\mu}_{teor}$' )#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,N0*e**((beta_R-beta_S)*T)*(e**((beta_R-beta_S)*T)-1)*((beta_R+beta_S)/(beta_R-beta_S)),color='r',ls=':',alpha = 0.95,label=r'$\sigma_{teor}$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,N0*e**((-beta_S)*T),color='k',ls=':',alpha = 0.95,label=r'$\bar{\mu}_{teor}$' )#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(T,N0*e**((-beta_S)*T)*(1-e**((-beta_S)*T)),color='r',ls=':',alpha = 0.95,label=r'$\sigma_{teor}$')#+'{:.{}f}'.format(SHfi, 3 ))


#    plt.plot(T,Lt+Zt,color=crta[j],alpha = 0.95,label=r'skupaj $\Delta t=$'+str(dt))#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
#plt.yscale('log')
plt.ylabel(' N(t) ')   
plt.xlabel(' t ')
#    plt.xlim([0,6])
#    plt.ylim([0,250])
#    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
plt.legend(loc=0)

############################ plotanje 3D mesh    
### nalogi a.) in b.)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#XX,TT = np.meshgrid(X,T)
#
#surf = ax.plot_wireframe(TT,XX,abs(M)**2, color = 'y', rstride=10, cstride=0, label='M=1')
#
##surf = ax.plot_wireframe(TT[0:150,:],XX[0:150,:],abs(M[0:150,:])**2, color = 'y', rstride=10, cstride=0, label='M=1')
#
##ax.plot_surface(TT, XX, M, rstride=8, cstride=8, alpha=0.3)
##cset = ax.contourf(TT, XX, M, zdir='m', offset=-100, cmap=cm.coolwarm)
##cset = ax.contourf(TT, XX, M, zdir='t', offset=-40, cmap=cm.coolwarm)
##cset = ax.contourf(TT, XX, M, zdir='y', offset=40, cmap=cm.coolwarm)
#ax.set_title( 'Časovni razvoj začetnega kvantnega stanja (n='+str(diff)+')')
#ax.set_xlabel("t")
#ax.set_ylabel("x")
#ax.set_zlabel("$|\psi(x,t)|^2$")
##ax.set_ylim([-20,20])
#ax.legend(loc='best')
#plt.tight_layout()
#plt.show()

