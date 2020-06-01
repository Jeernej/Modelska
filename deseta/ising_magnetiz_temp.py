# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:32:54 2019

@author: jernej
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

font = {
        'color':  'k',
        'size': 18,
        'verticalalignment': 'bottom'
        }
            
def avgE(s,j,h): 
    Eav=0
    for a in range(0,len(s)):        
        for b in range(0,len(s)):      

             # ciklični robni 
            if a==Dimenzija-1: 
                skplus1=s[0,b] 
            else: 
                skplus1=s[a+1, b]
	
            if b==Dimenzija-1: 
                slplus1=s[a,0] 
            else: 
                slplus1=s[a, b+1]

            if a==0: 
                skminus1=s[Dimenzija-1,b] 
            else: 
                skminus1=s[a-1, b]
	
            if b==0: 
                slminus1=s[a,Dimenzija-1] 
            else: 
                slminus1=s[a, b-1]
                
            # računanje avgE 
            Eav=Eav - j*s[a,b]*(skminus1 + skplus1 + slminus1 + slplus1) - h*s[a,b]
            
    return Eav/len(s)**2.
          

def Mag(s): 
    M=0
    for a in range(0,len(s)):        
        for b in range(0,len(s)):      
                                    
            M=M+s[a,b]
            
    return M/len(s)**2. 
    
     
def avgEE(s,j,h): 
    EEav=0
    for a in range(0,len(s)):        
        for b in range(0,len(s)):      
                        
             # ciklični robni 
            if a==Dimenzija-1: 
                skplus1=s[0,b] 
            else: 
                skplus1=s[a+1, b]
	
            if b==Dimenzija-1: 
                slplus1=s[a,0] 
            else: 
                slplus1=s[a, b+1]

            if a==0: 
                skminus1=s[Dimenzija-1,b] 
            else: 
                skminus1=s[a-1, b]
	
            if b==0: 
                slminus1=s[a,Dimenzija-1] 
            else: 
                slminus1=s[a, b-1]
                
            # računanje avgE 
            EEav=EEav + (-j*s[a,b]*(skminus1 + skplus1 + slminus1 + slplus1) - h*s[a,b])**2.
            
    return EEav/len(s)**2.
          

def MMag(s): 
    MM=0
    for a in range(0,len(s)):        
        for b in range(0,len(s)):      
                                    
            MM=MM+(s[a,b])**2.
            
    return MM/len(s)**2.
        
crta=['c','m','y','b','r','g','k']


for zz in range(1,5) :
#    Dimenzija=128
    Dimenzija=2**(3+zz)

    j=1
#    j=-1
    
    h=0.00001
#    h=0.00001+(zz-1)*0.1
    
    temp = 0.0001
    N=10**(5)
    
    #s1 = np.random.randint(2, size=(Dimenzija, Dimenzija))
    s1=np.ones((Dimenzija,Dimenzija))
    s2 = s1 - 1
    s = s2 + s1
    
    M=np.empty(0)
    E=np.empty(0)
    MM=np.empty(0)
    EE=np.empty(0)
    
    SuS=np.empty(0)
    Cv=np.empty(0)
    
    TEMP=np.empty(0)
    
#    F1=plt.figure(10)
#    F1=plt.subplot(2, 2, 1 ) 
#    plt.imshow(s2+s1,cmap=colors.ListedColormap(['white', '#72549A']))
#    plt.title('Mreža naključno urejenih spinov',fontdict=font)
    #plt.legend(loc='best')
    
    for k in range(0,100):
        
        for i in range(0,int(N)):
        	
            xrand = np.random.randint(0,len(s))
            yrand = np.random.randint(0,len(s))
            	
             # ciklični robni 
            if xrand==Dimenzija-1: skplus1=s[0,yrand] 
            else: skplus1=s[xrand+1, yrand]
            	
            if yrand==Dimenzija-1: slplus1=s[xrand,0] 
            else: slplus1=s[xrand, yrand+1]
            	
            if xrand==0: skminus1=s[Dimenzija-1,yrand] 
            else: skminus1=s[xrand-1, yrand]
            	
            if yrand==0: slminus1=s[xrand,Dimenzija-1] 
            else: slminus1=s[xrand, yrand-1]
             
             # računanje dE 
            deltaE = 2*j*s[xrand,yrand]*(skminus1 + skplus1 + slminus1 + slplus1) + 2*h*s[xrand,yrand]
          	
            if deltaE <= 0: 
             s[xrand,yrand] = -s[xrand,yrand]
            elif np.random.random() <= np.exp(-deltaE/temp): 
             s[xrand,yrand] = -s[xrand,yrand]
               
        
#        if k==5 :
#            F1=plt.subplot(2, 2, 2 ) 
#            plt.imshow(s,cmap=colors.ListedColormap(['white', '#72549A']))
#            plt.title(' T='+'{:.{}f}'.format(temp, 2 ) )
#        elif k==50:
#            F1=plt.subplot(2, 2, 3 ) 
#            plt.imshow(s,cmap=colors.ListedColormap(['white', '#72549A']))
#            plt.title(' T='+'{:.{}f}'.format(temp, 2 ) )
#        elif k==99:
#            F1=plt.subplot(2, 2, 4 ) 
#            plt.imshow(s,cmap=colors.ListedColormap(['white', '#72549A']))
#            plt.title(' T='+'{:.{}f}'.format(temp, 2 ) )
#                
#               
#                
        M=np.append(M,Mag(s))
        E=np.append(E,avgE(s,j,h))
        MM=np.append(MM,MMag(s))
        EE=np.append(EE,avgEE(s,j,h))
        
        temp=temp+0.075
        TEMP=np.append(TEMP,temp)
            
        print(k)
    
    #plt.suptitle('Mreža spinov po N='+str(N)+' potezah pri H='+str(h)+' in J='+str(j), fontsize=20)
    
#    
#    Tcx, Tcy = [2.269185*j, 2.269185*j], [-1, 1]
#    XX, YY = [0, 5], [0, 0]
#    
#    plt.suptitle('N='+str(N)+', H='+str(h)+', J='+str(j), fontsize=20)

    F10=plt.figure(9)
    
    F10=plt.subplot(2, 2, 1 )  
    plt.plot(TEMP,M,color=crta[zz],alpha = 0.95,label=r'$x=$'+str(Dimenzija))
#    plt.plot(TEMP,M,'k',alpha = 0.95,label=r'$\langle M \rangle$') label=r'$\chi$'
    #plt.plot(TEMP,(M**2/len(s)**2.-(M/len(s))**2)/TEMP,'r:',alpha = 0.95,label=r'$\chi$')
    #plt.plot(TEMP,((MM-M**2)/len(s))**2/TEMP,'r:',alpha = 0.95,label=r'$\chi$')
    #plt.plot(Tcx, Tcy, "k:",alpha=0.6,label=r'$T_c=2.269185.. \frac{J}{k_B}$')
    #plt.plot(XX, YY, "k")
    plt.axvline(x=2.269185,color='k', linestyle=':')
    plt.xlabel('T',fontsize=16)
    #plt.ylabel('sprejete poteze',fontsize=16)
    #plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('Magnetizacija \n'+'( J='+str(j)+', H='+'{:.{}f}'.format(h, 2 )+', N='+str(N)+')' ,fontdict=font)
    plt.legend(loc='best',fontsize=16)
    
    F10=plt.subplot(2, 2, 2 )  
    plt.plot(TEMP,((MM-M**2)/len(s))**2/TEMP,color=crta[zz],alpha = 0.95,label=r'$x=$'+str(Dimenzija))
#    plt.plot(TEMP,((MM-M**2)/len(s))**2/TEMP,'r:',alpha = 0.95,label=r'$\chi$')
    #plt.plot(Tcx, Tcy, "k:",alpha=0.6,label=r'$T_c=2.269185.. \frac{J}{k_B}$')
    #plt.plot(XX, YY, "k")
    plt.axvline(x=2.269185,color='k', linestyle=':')
    plt.xlabel('T',fontsize=16)
    #plt.ylabel('sprejete poteze',fontsize=16)
#    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('Susceptibilnost \n'+'( J='+str(j)+', H='+'{:.{}f}'.format(h, 2 )+', N='+str(N)+')' ,fontdict=font)
    plt.legend(loc='best',fontsize=16)
    
       
    F10=plt.subplot(2, 2, 3 )  
    plt.plot(TEMP,E,color=crta[zz],alpha = 0.95,label=r'$x=$'+str(Dimenzija))
#    plt.plot(TEMP,E,'c',alpha = 0.95,label=r'$\langle E\rangle$')    
    #plt.plot(TEMP,(E**2/len(s)**2.-(E/len(s))**2)/TEMP**2,'m',alpha = 0.95,label=r'$C_v$')
    #plt.plot(TEMP,(EE-E**2)/(len(s)*TEMP)**2,'m',alpha = 0.95,label=r'$C_v$')
    plt.axvline(x=2.269185,color='k', linestyle=':')
    plt.xlabel('T',fontsize=16)
    #plt.ylabel(r'$E$',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    #plt.xscale('log')
    plt.title('Energija \n'+'( J='+str(j)+', H='+'{:.{}f}'.format(h, 2 )+', N='+str(N)+')' ,fontdict=font)
    plt.legend(loc='best',fontsize=16)
    
    
    F10=plt.subplot(2, 2, 4 )  
    #plt.plot(TEMP,E,'c',alpha = 0.95,label=r'$\langle E\rangle$')
    #plt.plot(TEMP,(E**2/len(s)**2.-(E/len(s))**2)/TEMP**2,'m',alpha = 0.95,label=r'$C_v$')
    plt.plot(TEMP,(EE-E**2)/(len(s)*TEMP)**2,color=crta[zz],alpha = 0.95,label=r'$x=$'+str(Dimenzija))
#    plt.plot(TEMP,(EE-E**2)/(len(s)*TEMP)**2,'m',alpha = 0.95,label=r'$C_v$')
    plt.axvline(x=2.269185,color='k', linestyle=':')
    plt.xlabel('T',fontsize=16)
    #plt.ylabel(r'$E$',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
#    plt.yscale('log')
    plt.title('Specifična toplota  \n'+'( J='+str(j)+', H='+'{:.{}f}'.format(h, 2 ) +', N='+str(N)+')' ,fontdict=font)
    plt.legend(loc='best',fontsize=16)




