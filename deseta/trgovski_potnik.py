import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import colors

#np.random.seed(0)
#fig = plt.figure(figsize=(12,10)) 
#ax =fig.add_subplot(111)

#temp = 0.001

N=10
x=np.random.random(N)
y=np.random.random(N)

print(x)
print(y)

#ax.plot(x,y, 'ro')

L1=0
L2=0
home=np.random.randint(N)
xhome = x[home]
yhome = y[home]


ind = np.arange(N)
np.random.shuffle(ind)
print(ind)

quit()

for i in range(10):
	
	ind = np.arange(N)
	np.random.shuffle(ind)

	L=0.
	for j in range(len(ind)-1):
		L += np.sqrt( (x[ind[j]]-x[ind[j+1]])**2 + (y[ind[j]] - y[ind[j+1]])**2 )
	
	#L1 += np.sqrt((x) 
	
	#L1 = np.sqrt((xy[-x1)**2 + (yhome-y1)**2) + np.sqrt((x2-x1)**2 + (y2-y1)**2)
	#L2 = np.sqrt((xhome-x2)**2 + (yhome-y2)**2) + np.sqrt((x2-x1)**2 + (y2-y1)**2)
	
	#deltaE = L2-L1
		
	#if deltaE <= 0: 
	#	xhome=x1
	#	yhome=y1
	#elif np.random.random() <= np.exp(-deltaE/temp):
	#	xhome=x1
	#	yhome=y1		

		
#plt.show()

F10=plt.figure(9)

#F10=plt.subplot(2, 2, 1 )  
plt.plot(x,y,color='ko',alpha = 0.95,label=r'$N=$'+str(N))
#    plt.plot(TEMP,M,'k',alpha = 0.95,label=r'$\langle M \rangle$') label=r'$\chi$'
#plt.plot(TEMP,(M**2/len(s)**2.-(M/len(s))**2)/TEMP,'r:',alpha = 0.95,label=r'$\chi$')
#plt.plot(TEMP,((MM-M**2)/len(s))**2/TEMP,'r:',alpha = 0.95,label=r'$\chi$')
#plt.plot(Tcx, Tcy, "k:",alpha=0.6,label=r'$T_c=2.269185.. \frac{J}{k_B}$')
#plt.plot(XX, YY, "k")
#plt.axvline(x=2.269185,color='k', linestyle=':')
#plt.xlabel('T',fontsize=16)
#plt.ylabel('sprejete poteze',fontsize=16)
#plt.xscale('log')
#plt.tick_params(axis='both', which='major', labelsize=16)
#plt.title('Magnetizacija \n'+'( J='+str(j)+', H='+'{:.{}f}'.format(h, 2 )+', N='+str(N)+')' ,fontdict=font)
#plt.legend(loc='best',fontsize=16)
    
