import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)


N=100
x=np.random.random(N)
y=np.random.random(N)

x=list(x)
y=list(y)

#print(x,y)

x.append(x[0]) #doda prvo tocko kot zadnjo, da se vrne v cilj
y.append(y[0]) #doda prvo tocko kot zadnjo, da se vrne v cilj
rand_dolzina=0
#print(x,y)
for j in range(0,N): 
    rand_dolzina+=np.sqrt( (x[j]-x[j+1])**2 + (y[j]-y[j+1])**2 )
    
F10=plt.figure(9)
F10=plt.subplot(1, 2, 1 )  
plt.plot(x,y, 'ro',alpha = 0.95,ls='--')
plt.title('Pot trgovca skozi N='+str(N)+' mest po naključni poti dolžine'+' {:.{}f}'.format(rand_dolzina, 2 ))

temp = 0.00000001
#temp = 2

npotez = 0
dolzina=0
for k in range(0,100*N):

    L1=0

    for j in range(0,N): L1+=np.sqrt( (x[j]-x[j+1])**2 + (y[j]-y[j+1])**2 )

#    ind = np.random.random_sample((np.arange(1,N),2)) #izbere dva razlicna random indeksa od 1 do N-1 
    ind=np.random.randint(1,N, size=(2, 1))
    
    xnew=list(x) #kopiranje listov
    ynew=list(y) #kopiranje listov

    xnew[ind[0][0]],xnew[ind[1][0]]=xnew[ind[1][0]],xnew[ind[0][0]] #zamenja x (med indeksoma ind)
    ynew[ind[0][0]],ynew[ind[1][0]]=ynew[ind[1][0]],ynew[ind[0][0]] #zamenja y (med indeksoma ind)
    
    L2=0
    
    for j in range(0,N): L2+=np.sqrt( (xnew[j]-xnew[j+1])**2 + (ynew[j]-ynew[j+1])**2 )

#
#    if (0.4<xnew[ind[0][0]-1]<0.6 or 0.4<xnew[ind[0][0]+1]<0.6) and (xnew[ind[1][0]]<0.4 or xnew[ind[1][0]]>0.6):
#        if (0.4<ynew[ind[0][0]-1]<0.6 or 0.4<ynew[ind[0][0]+1]<0.6) and (ynew[ind[1][0]]<0.4 or ynew[ind[1][0]]>0.6):
#            L2=L2+0.05
#
#    elif 0.4<xnew[ind[1][0]]<0.6 and (xnew[ind[0][0]]<0.4 or xnew[ind[0][0]]>0.6):
#        if 0.4<ynew[ind[1][0]]<0.6 and (ynew[ind[0][0]]<0.4 or ynew[ind[0][0]]>0.6):
#            L2=L2+0.05

    deltaE = L2-L1

    if deltaE <= 0: 
        x=list(xnew)
        y=list(ynew)
        npotez+=1
    elif np.random.random() <= np.exp(-deltaE/temp):
        x=list(xnew)
        y=list(ynew)
        npotez+=1 


print(npotez)  
print(dolzina)            
          
#print(x,y)


F10=plt.subplot(1, 2, 2 )  
plt.plot(x,y, 'ro',alpha = 0.95,ls='--')
plt.title('Pot trgovca skozi N='+str(N)+' mest po najkrajši poti dolžine'+' {:.{}f}'.format(L1, 2 ))
  
    


