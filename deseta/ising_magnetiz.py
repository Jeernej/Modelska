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
                        
            if a==len(s)-1: 
                skplus1=s[0,b] 
            else: 
                skplus1=s[a+1, b]
	
            if b==len(s)-1: 
                slplus1=s[a,0] 
            else: 
                slplus1=s[a, b+1]
            
            Eav=Eav -j*s[a,b]*(s[a-1, b] + skplus1 + s[a,b-1] + slplus1) + h*s[a,b]
            
    return Eav/len(s)**2.
          

def Mag(s): 
    M=0
    for a in range(0,len(s)):        
        for b in range(0,len(s)):      
                                    
            M=M+s[a,b]
            
    return M/len(s)**2. 
    
    

        
Dimenzija=20
j=1.
h=0.0
temp = 0.00001
N=10**(7.)

s1 = np.random.randint(2, size=(Dimenzija, Dimenzija))
s2 = s1 - 1
s = s2 + s1

M=np.empty(0)
E=np.empty(0)
SuS=np.empty(0)
Cv=np.empty(0)
TEMP=np.empty(0)

F10=plt.figure(10)
for k in range(0,7):
    
    for i in range(0,int(N)):
    	
    	xrand = np.random.randint(0,len(s))
    	yrand = np.random.randint(0,len(s))
    	
    	if xrand==len(s)-1: skplus1=s[0,yrand] 
    	else: skplus1=s[xrand+1, yrand]
    	
    	if yrand==len(s)-1: slplus1=s[xrand,0] 
    	else: slplus1=s[xrand, yrand+1]
    	
    	deltaE = 2*j*s[xrand,yrand]*(s[xrand-1, yrand] + skplus1 + s[xrand,yrand-1] + slplus1) + 2*h*s[xrand,yrand]
    		
    	if deltaE <= 0: 
         s[xrand,yrand] = -s[xrand,yrand]
    	elif np.random.random() <= np.exp(-deltaE/temp): 
         s[xrand,yrand] = -s[xrand,yrand]
    
    if k==0 or k==2 or k==4 or k==6:
        F10=plt.subplot(2, 2, k-1 ) 
        #plt.imshow(s2+s1,cmap='Greys')
        plt.imshow(s2+s1,cmap=colors.ListedColormap(['white', '#72549A']))
        plt.title('Kvadratna mreža naključno razporejenih spinov')

    M=np.append(M,Mag(s))
    E=np.append(E,avgE(s,j,h))
    
    temp=temp+0.75
    TEMP=np.append(TEMP,temp)




F10=plt.subplot(1, 2, 2 ) 
plt.imshow(s,cmap=colors.ListedColormap(['white', '#72549A']))
plt.title('Mreža spinov po N='+str(N)+' potezah pri temperaturi T='+str(temp)+ ', H='+str(h)+'in J='+str(j))
#plt.legend(loc='best')


Tcx, Tcy = [2.269185*j, 2.269185*j], [0, 1]

F10=plt.figure(9)
F10=plt.subplot(1, 2, 1 )  
plt.plot(TEMP,M,'k',alpha = 0.95,label=r'$\langle M \rangle$')
plt.plot(TEMP,(M**2/len(s)**2.-M**2)/TEMP,'r:',alpha = 0.5,label=r'$\chi$')
plt.plot(Tcx, Tcy, "k:",alpha=0.6,label=r'$T_c=2.269185 \frac{J}{k_B}$')
plt.xlabel('T',fontsize=16)
#plt.ylabel('sprejete poteze',fontsize=16)
#plt.xscale('log')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.title('Magnetizacija in susceptibilnost  '+'( J='+str(j)+'H='+str(h)+'N='+str(N)+')' ,fontdict=font)
plt.legend(loc='best',fontsize=16)
   
F10=plt.subplot(1, 2, 2 )  
plt.plot(TEMP,E,'c',alpha = 0.95,label=r'$\langle E\rangle$')
plt.plot(TEMP,(E**2/len(s)**2.-E**2)/TEMP**2,'m',alpha = 0.95,label=r'$C_v$')
plt.xlabel('T',fontsize=16)
#plt.ylabel(r'$E$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
#plt.xscale('log')
plt.title('Energija in specifična toplota  '+'( J='+str(j)+'H='+str(h)+'N='+str(N)+')' ,fontdict=font)
plt.legend(loc='best',fontsize=16)



