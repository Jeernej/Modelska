# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import math 
import numpy as np
#import scipy as sc
import matplotlib.pyplot  as plt
#from scipy import stats
#from scipy.optimize import curve_fit
from lmfit import Model
from numpy import loadtxt
from scipy.optimize import curve_fit

        
e=2.718281828459045    
pi=3.141592653589793 


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sesta/thtg-xfp-thfp.dat'

###_________Modeli__________________________________________ 
    

def gen_X(x,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,x**(i+1))
    return X

def gen_XinY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x+y)**(i+1))
    return X
    
def gen_XminusY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x-y)**(i+1))
    return X
    
def gen_XkratY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x*y)**(i+1))
    return X

def gen_XdeljY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x/y)**(i+1))
    return X
    
def gen_YdeljX(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(y/x)**(i+1))
    return X

def gen_cosXsinY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(np.cos(x)+np.sin(y))**(i+1))
    return X
    
def gen_xcosY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x*np.cos(y))**(i+1))
    return X    
    
def gen_YsinX(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(y*np.sin(x))**(i+1))
    return X 

#def ctgXtgY(x,y,M):
#    X=np.empty(0)
#    for i in range(M): 
##        X=np.append(X,(np.tan(y)*np.tan(x))**(1/(i+1)))
#        X=np.append(X,abs(y+x)**(-(i+1)))
#    return X 

    
###_________Fit_funkcije__________________________________________ 

    
#def gen_A(x,fi,sig_x,sig_fi,M):
#    A=np.empty([len(x),3*M+1])  ## M mniožim s številom odkomentiranih členov  !!!
#    for i in range(len(x)):        #posamezni členi
#        X=gen_X(x[i],M) ## potence x^i
#        Fi=gen_X(fi[i],M) ## potence fi^i 
#        xinfi=gen_XinY(x[i],fi[i],M) # potence (x+fi)^i
##        xmanjfi=gen_XminusY(x[i],fi[i],M) # potence (x-fi)^i
##        xkratfi=gen_XkratY(x[i],fi[i],M)  # potence (x*fi)^i
##        xdelfi=gen_XdeljY(x[i],fi[i],M)  # potence (x/fi)^i
##        fidelx=gen_YdeljX(x[i],fi[i],M) # potence (fi/x)^i
#        
##        vrstica=np.concatenate((np.ones([1]),X,Fi,xinfi,xmanjfi,xkratfi,xdelfi,fidelx), axis=None)
#        vrstica=np.concatenate( ( 1/(sig_x[0]*sig_fi[0]) , X/sig_x[0] , Fi/sig_fi[0], xinfi/(sig_x[0]*sig_fi[0]) ) , axis=None)
#
#        A[i]=vrstica   ###/np.sqrt(sig_x[0]**2+sig_fi[0]**2)
#        
#    return A  # matrika  modela za vsako meritev x_i in fi_i

def gen_A(x,fi,sig_x,M):
    A=np.empty([len(x),6*M+1])  ## M mniožim s številom odkomentiranih členov  !!!
    for i in range(len(x)):        #posamezni členi
        X=gen_X(x[i],M) ## potence x^i
        Fi=gen_X(fi[i],M) ## potence fi^i 
        xinfi=gen_XinY(x[i],fi[i],M) # potence (x+fi)^i
        xmanjfi=gen_XminusY(x[i],fi[i],M) # potence (x-fi)^i
        xkratfi=gen_XkratY(x[i],fi[i],M)  # potence (x*fi)^i
        fidelx=gen_YdeljX(x[i],fi[i],M) # potence (fi/x)^i
#        CxSy=gen_cosXsinY(x[i],fi[i],M) # potence (fi/x)^i
#        xCy=gen_xcosY(x[i],fi[i],M) # potence (fi/x)^i
#        ySx=gen_YsinX(x[i],fi[i],M) # potence (fi/x)^i
#        CxCTy=ctgXtgY(x[i],fi[i],M) # potence (fi/x)^i

#        vrstica=np.concatenate((np.ones([1]),X,Fi,xinfi,xmanjfi,xkratfi,xdelfi,fidelx), axis=None)
        vrstica=np.concatenate((np.ones([1]),X,Fi,xinfi,xmanjfi,xkratfi,fidelx), axis=None)
#        vrstica=np.concatenate( ( np.ones([1]),X,Fi,xinfi,xmanjfi,xkratfi ) , axis=None)

        A[i]=vrstica/sig_x[0]    ###/np.sqrt(sig_x[0]**2+sig_fi[0]**2)
        
    return A  # matrika  modela za vsako meritev x_i in fi_i
    
    
def gen_a(U,S,V,b):
    a=0
    for i in range(len(V)):
        a=a+sum(U[:,i]*b)*V[i]/S[i]
    return a 

def sig2(S,V):   
    sigma2=np.zeros([len(V)])
    for j in range(len(V)):
        for i in range(len(V)):        
            sigma2[j]=sigma2[j]+(V[j,i]/S[i])**2
    return sigma2


### preverjanje rešitev

def model(x,fi,a,M):  
    A=np.ones([1])
    mod=np.empty([0])
    for i in range(len(x)):        
        X=gen_X(x[i],M)  ## potence x^i
        Fi=gen_X(fi[i],M) ## potence fi^i
        xinfi=gen_XinY(x[i],fi[i],M)  # potence (x+fi)^i
        xmanjfi=gen_XminusY(x[i],fi[i],M)  # potence (x-fi)^i
        xkratfi=gen_XkratY(x[i],fi[i],M) # potence (x*fi)^i
        fidelx=gen_YdeljX(x[i],fi[i],M)  # potence (fi/x)^i
#        CxSy=gen_cosXsinY(x[i],fi[i],M) # potence (fi/x)^i
#        xCy=gen_xcosY(x[i],fi[i],M) # potence (fi/x)^i
#        ySx=gen_YsinX(x[i],fi[i],M) # potence (fi/x)^i
#        CxCTy=ctgXtgY(x[i],fi[i],M) # potence (fi/x)^i
       
#        A=np.concatenate((A,X,Fi,xinfi,xmanjfi,xkratfi,xdelfi,fidelx), axis=None)      
        A=np.concatenate((A,X,Fi,xinfi,xmanjfi,xkratfi,fidelx), axis=None)      
#        A=np.concatenate((A,X,Fi,xinfi,xmanjfi,xkratfi), axis=None)      

        mod=np.append(mod,sum(a*A))        
        A=np.ones([1])
        
    return mod  # vektor izračuna th po modelu za vsake meritve x_i in fi_i
    
    
def chi2(th,x,fi,a,sig_th):
    M=len(a)
    N=len(th)
    chi=sum(((th-model(x,fi,a))/sig_th)**2)    
    return chi/(N-M)


def Hdata(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins='auto',normed=False)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram, polažaji sredine binov, število binov
     
###_________podatki______________________________________________________________________________________ 


data = loadtxt(DIR) # branje

## meritve
FI_fp=data[:,0]

for i in range(0,len(FI_fp)) : FI_fp[i]=math.radians(FI_fp[i])
X_fp=data[:,1]

TH_tg=data[:,2]
for i in range(0,len(TH_tg)) : TH_tg[i]=math.radians(TH_tg[i])

## negotovosti meritev

sig_FI_fp=np.ones(len(FI_fp))/1000
sig_X_fp=np.ones(len(X_fp))
sig_TH_tg=np.ones(len(TH_tg))/1000

### računanje modelskih parametrov ______________________________________________________________________________________ 

N=len(X_fp)
M=6
m=1+M*6  ## M mniožim s številom odkomentiranih členov v gen_A() !!!

b=TH_tg/sig_TH_tg
#A=gen_A(X_fp,FI_fp,sig_X_fp,sig_FI_fp,M)
A=gen_A(X_fp,FI_fp,sig_TH_tg,M)
#print(np.shape(A))
u, s, vh = np.linalg.svd(A, full_matrices=False)

a=gen_a(u, s, vh ,b) ## optimalni parametri
print('parametri:')
print(a)
sig2_a=sig2(s,vh)  ## negotovosti optimalnih parametrov
print('negotovosti parametrov:')
print(sig2_a)


#### chi2 test in histogram porazdelitev odstopanj modela od merjenih kotov na tarči ___________________________________________ 

TH_tg_model=model(X_fp,FI_fp,a,M)
chi=sum(((TH_tg-TH_tg_model)/sig_TH_tg)**2)/(N-len(a))
Hfi, xfi, binEdge_fi, Lfi=Hdata(TH_tg-TH_tg_model)
#
#c=0
#for i in range(0,len(a)) : 
#    if np.log10(abs(a[i]))<10**(-6): 
#        a[i]=0
#        c=c+1
#
#TH_tg_modelCUT=model(X_fp,FI_fp,a,M)
#chi_CUT=sum(((TH_tg-TH_tg_modelCUT)/sig_TH_tg)**2)/(N-len(a)-c)
#Hfi_CUT, xfi_CUT, binEdge_fi_CUT, Lfi_CUT=Hdata(TH_tg-TH_tg_modelCUT)

F10=plt.figure(10)
F10=plt.subplot(1,2, 1 )  
plt.title('Histogram porazdelitev odstopanj vrednosti modela od merjenih vrednosti',fontsize=16)
plt.step(xfi,Hfi,'k',alpha = 0.5,label=r'$\chi^2$='+'{:.{}f}'.format(chi,1))
#plt.step(xfi_CUT,Hfi_CUT,'m',alpha = 0.5,label=r'odstopanja;  pri $\chi_{cut}^2$='+str(chi_CUT))
#plt.step(Y-Model_Fdata(X,y0,a),'c',alpha = 0.5,label=r'odstopanja')
plt.xlabel(r'$ y_i-\widetilde{y}_i $')
plt.ylabel('n')
plt.legend(loc='best')

x=np.linspace(1,len(a),len(a))
F10=plt.subplot(1,2, 2 )  
plt.title('Absolutne vrednosti optimalnih parametrov',fontsize=16)
plt.errorbar(x,abs(a), yerr=sig2_a,color='k',alpha = 0.95)#,label=r' vrednosti parametrov;  pri $\chi^2$='+str(chi))
#plt.step(Y-Model_Fdata(X,y0,a),'c',alpha = 0.5,label=r'odstopanja')
plt.xlabel(r'$i $')
plt.ylabel(r'$a_i$')
plt.yscale('log')
plt.legend(loc='best')

###_____ploti____________

#x=np.linspace(1,len(FI_fp),len(FI_fp))
#
#F10=plt.figure(3)
#plt.title('Kalibracijski podatki visokoločljivostnega magnetnega spektrometra',fontsize=16)
#plt.errorbar(x,FI_fp, yerr=sig_FI_fp, color='r', marker='o',label=r'$ \vartheta_{tg} $ [rad]',markersize=2)
#plt.errorbar(x,X_fp, yerr=sig_X_fp, color='g', marker='o',label=r'$ x_{fp} $ [mm]',markersize=2)
#plt.errorbar(x,TH_tg, yerr=sig_TH_tg, color='b', marker='o',label=r'$ \vartheta_{fp}$ [rad]',markersize=2)
#plt.xlabel(r'$št. meritve$',fontsize=16)
#plt.legend(loc=0)
#