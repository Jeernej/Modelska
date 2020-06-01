# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:15:15 2019

@author: jernej
"""
import numpy as np
import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize

    
e=2.718281828459045    
pi=3.141592653589793 
e0=8.8541878128*10**(-12)

   
def energija(zacetni): 
    N=int(len(zacetni)/2)   ## <-- !! popravi za vsak N
    en=0
    for i in range(0,N-1):
#        print('i='+str(i))
        for j in range(i+1,N):
#            en = en + np.sqrt( 2 - 2 * ( np.cos(zacetni[i])*np.cos(zacetni[j])  +  np.sin(zacetni[j])*np.sin(zacetni[i]) * np.sin(zacetni[i+N]) * np.sin(zacetni[j+N]) + np.sin(zacetni[j])*np.sin(zacetni[i]) * np.cos(zacetni[i+N]) * np.cos(zacetni[j+N]) ))**(-1)
            en = en + np.sqrt( 2 - 2 * ( np.cos(zacetni[i])*np.cos(zacetni[j])  +  np.sin(zacetni[j]) * np.sin(zacetni[i]) * np.sin(zacetni[i+N]) * np.sin(zacetni[j+N]) + np.sin(zacetni[j])*np.sin(zacetni[i]) * np.cos(zacetni[i+N]) * np.cos(zacetni[j+N]) ))**(-1)
#            print('j='+str(j))
#            print('en='+str(en))
    return en

###### ___zaćetne rednosti

N=40 ## !! popravi tudi zgoraj v zanki energija !!!

th=pi*np.random.rand(N)
fi=np.random.rand(N)*pi*2
#zacetni=[th,fi]    ## zapakirani argumenti v eno spremenljivko
zacetni=np.append(th,fi)

###### ___minimizacija

Rl_Pot_En=energija(zacetni)
print(Rl_Pot_En)


#res1 = minimize(energija, zacetni, method='Nelder-Mead', tol=1e-9, options={'maxiter': 5000}) ## 'Nelder-Mead 
#print(res1)
#res2 = minimize(energija, zacetni, method='Powell', tol=1e-9, options={'maxiter': 5000}) ## 'Nelder-Mead 
#print(res2)
res3 = minimize(energija, zacetni, method='L-BFGS-B', tol=1e-9, options={'maxiter': 5000}) ## 'Nelder-Mead 
print(res3)

#Rl_Pot_En_min1=energija(res1.x)
#Rl_Pot_En_min2=energija(res2.x)
Rl_Pot_En_min3=energija(res3.x)

#print(Rl_Pot_En_min1)
#print(Rl_Pot_En_min2)
print(Rl_Pot_En_min3)


####### ____rezultati_v _vektorjih________________________
##
NN=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,28,40])
wiki=np.array([0.500000000,1.732050808,3.674234614,6.474691495,9.985281374,14.452977414,19.675287861,25.759986531,32.716949460,40.596450510,49.165253058,58.853230612,69.306363297,80.670244114,92.911655302,106.050404829,120.084467447,135.089467557,150.881568334,310.491542358,660.675278835])
RES_N_M=np.array([0.500000000,1.732050808,3.674234614,6.474691495,9.985281374,14.457944046,19.677913334,25.760690236,32.717269766,40.605005689,49.171186795,59.106257411,69.423245036,80.672985750,93.764705112,107.462677300,120.097872936,142.637369424,151.137590246,314.331843195,681.766775818])

RES_PW=np.array([0.500000000,1.732050808,3.674234614,6.474691495,9.985281375,14.452977717,19.675287864,25.759986532,32.716949471,40.596450515,49.165253094,58.853230762,69.306363334,80.670244562,92.911655495,106.050405390,120.084467913,135.089480581,150.881570326,310.491565091,660.705855311])

RES_L_BFGS_B=np.array([0.500000000,1.732050808,3.674234614,6.474691495,9.985281374,14.452977416,19.675287863,25.759986536,32.716949471,40.596450565,49.165253072,58.853230634,69.306363331,80.670244188,92.911655326,106.050405099,120.084467637,135.089468697,150.883370816,310.491543495,660.675281212])

Iter_N_M=np.array([132,250,447,680,1423,2105,2458,2774,3139,3529,3878,4278,4603,5000,5000,5000,5000,5000,5000,5000,5000])

Iter_PW=np.array([3,6,7,8,10,18,14,91,21,29,30,28,27,27,41,34,32,37,41,42,42])

Iter_L_BFGS_B=np.array([7,9,10,16,23,38,27,27,32,37,41,50,50,51,59,81,67,68,54,76,104])
#
#print(len(NN))
#print(len(wiki ))
#print(len(RES_N_M ))
#print(len(RES_PW ))
#print(len( RES_L_BFGS_B))
#len( )

####### ____
#
#
#F50=plt.figure(50)
#F50=plt.subplot(3, 1, 1 ) 
#plt.title(r'Izračuni minimizirane elektrostatske energije za različne $N$',fontsize=16)
#plt.plot(NN , wiki , color='k', ls='-', alpha = 0.95, label='Vir [2]')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN , RES_N_M ,color='r',ls='-',alpha = 0.95,label='Nelder-Mead')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN , RES_PW ,color='g',ls='-',alpha = 0.95,label='Powell')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN , RES_L_BFGS_B ,color='b',ls='-',alpha = 0.95,label='L-BFGS-B')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.ylabel(r'$V_N^{min}$' ,fontsize=16)   
#plt.xlabel(r'$N$' ,fontsize=16)
##plt.xlim([0,LS3/2])
#plt.legend(loc=0)
##plt.yscale('log')
#
#
##
#F50=plt.subplot(3, 1, 2 ) 
#plt.title(r'Odstopanja med vrednostmi Vir [2] in izračuni minimizirane elektrostatske energije za različne $N$',fontsize=16)
#plt.plot(NN[0:len(RES_N_M)] , abs(wiki[0:len(RES_N_M)]-RES_N_M) ,color='r',ls='-',alpha = 0.95,label='Nelder-Mead')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN[0:len(RES_PW)] , abs(wiki[0:len(RES_PW)]-RES_PW) ,color='g',ls='-',alpha = 0.95,label='Powell')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN[0:len(Iter_L_BFGS_B)] , abs(wiki[0:len(Iter_L_BFGS_B)]-RES_L_BFGS_B) ,color='b',ls='-',alpha = 0.95,label='L-BFGS-B')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.ylabel(r'$|\Delta V_N^{min}|$' ,fontsize=16)   
#plt.xlabel(r'$N$' ,fontsize=16)
##plt.xlim([0,LS3/2])
#plt.yscale('log')
#plt.legend(loc=0)
#
#
#
#F50=plt.subplot(3, 1, 3 ) 
#plt.title(r'Število izvedenih iteracij minimizacijske metode za različne $N$',fontsize=16)
#plt.plot(NN[0:len(Iter_N_M)] , Iter_N_M ,color='r',ls='-',alpha = 0.95,label='Nelder-Mead')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN[0:len(Iter_PW)] , Iter_PW ,color='g',ls='-',alpha = 0.95,label='Powell')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(NN[0:len(Iter_L_BFGS_B)] , Iter_L_BFGS_B ,color='b',ls='-',alpha = 0.95,label='L-BFGS-B')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.ylabel(r'$n_{it}$' ,fontsize=16)   
#plt.xlabel(r'$N$' ,fontsize=16)
##plt.xlim([0,LS3/2])
#plt.legend(loc=0)
#plt.yscale('log')
#



###### ____izris_sfer z naboji_______________________________

xN=np.sin(th)*np.cos(fi)
yN=np.sin(th)*np.sin(fi)
zN=np.cos(th)

F20=plt.figure(20)
#F20=plt.subplot(1, 2, 1 ) 
Axes3D = plt.axes(projection='3d')
Axes3D.scatter(xN, yN, zN, zdir='z',marker='o', s=10, c='r', depthshade=True)
plt.title('Naključna začetna porazdelitev elektronov po enotski krogli ;   N='+str(N))

# draw sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)

plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
#plt.zlim([-1.2,1.2])

###### ____

Th=res3.x[0:N]
Fi=res3.x[N:2*N]

XN=np.sin(Th)*np.cos(Fi)
YN=np.sin(Th)*np.sin(Fi)
ZN=np.cos(Th)

F10=plt.figure(10)
#F20=plt.subplot(1, 2, 2 ) 
Axes3D = plt.axes(projection='3d')
Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True)
plt.title('Porazdelitev elektronov po enotski krogli za minimalno energijo ;   N='+str(N))

# draw sphere
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)

plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
#plt.zlim([-1.2,1.2])
