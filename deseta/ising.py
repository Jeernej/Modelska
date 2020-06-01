import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#np.random.seed(0)
    
def avgE(s,j,h): 
    Eav=0
    for a in range(0,len(s)):        
        for b in range(0,len(s)):      

             # ciklični robni 
            if a==199: 
                skplus1=s[0,b] 
            else: 
                skplus1=s[a+1, b]
	
            if b==199: 
                slplus1=s[a,0] 
            else: 
                slplus1=s[a, b+1]

            if a==0: 
                skminus1=s[199,b] 
            else: 
                skminus1=s[a-1, b]
	
            if b==0: 
                slminus1=s[a,199] 
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
            if a==199: 
                skplus1=s[0,b] 
            else: 
                skplus1=s[a+1, b]
	
            if b==199: 
                slplus1=s[a,0] 
            else: 
                slplus1=s[a, b+1]

            if a==0: 
                skminus1=s[199,b] 
            else: 
                skminus1=s[a-1, b]
	
            if b==0: 
                slminus1=s[a,199] 
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

      
      
j=-1
h=0
h=1.8
temp = 0.00002
temp = 2.269185
temp = 3
N=10**(6)

s1 = np.random.randint(2, size=(200, 200))  #spini poz
#s1=-np.ones((200,200))

s2 = s1 - 1  #spini neg
s = s2 + s1  # spini poz in neg

for i in range(0,int(N)):
	
	xrand = np.random.randint(0,200)
	yrand = np.random.randint(0,200)
	
     # ciklični robni 
	if xrand==199: skplus1=s[0,yrand] 
	else: skplus1=s[xrand+1, yrand]
	
	if yrand==199: slplus1=s[xrand,0] 
	else: slplus1=s[xrand, yrand+1]
	
	if xrand==0: skminus1=s[199,yrand] 
	else: skminus1=s[xrand-1, yrand]
	
	if yrand==0: slminus1=s[xrand,199] 
	else: slminus1=s[xrand, yrand-1]
     
     # računanje dE 
	deltaE = 2*j*s[xrand,yrand]*(skminus1 + skplus1 + slminus1 + slplus1) + 2*h*s[xrand,yrand]
		
	if deltaE <= 0: s[xrand,yrand] = -s[xrand,yrand]  #pogoj Za spin flip
	elif np.random.random() <= np.exp(-deltaE/temp): s[xrand,yrand] = -s[xrand,yrand]  #pogoj Za spin flip

    

F10=plt.figure(10)
##F10=plt.subplot(1, 2, 1 ) 
##plt.imshow(s2+s1,cmap='Greys')
#plt.imshow(s2+s1,cmap=colors.ListedColormap(['white', '#72549A']))
#plt.title('Kvadratna mreža naključno razporejenih spinov')

#F10=plt.subplot(1, 2, 2 ) 
plt.imshow(s,cmap=colors.ListedColormap(['white', '#72549A']))
plt.title(r'Mreža spinov po N='+str(N)+' potezah \n pri temperaturi T='+'{:.{}f}'.format(temp, 2 )+ ', H='+str(h)+' in J='+str(j))
#plt.legend(loc='best')
    

    
