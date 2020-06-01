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
from scipy import stats

e=2.718281828459045    
pi=3.141592653589793 

DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sesta/thtg-xfp-thfp.dat'


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
  
  
     
###_________Fit_funkcije__________________________________________ 
    

def gen_X(x,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,x**i)
#        print(i)
    return X

def gen_XinY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x+y)**i)
    return X
    
def gen_XkratY(x,y,M):
    X=np.empty(0)
    for i in range(M): 
        X=np.append(X,(x*y)**i)
    return X

   
def gen_A(x,sig,M):
    A=np.zeros([len(x),M])
    for i in range(len(x)):
        fi=gen_X(x[i],M)
#        print(fi)
        for j in range(M):
            A[i,j]=fi[j]/sig[i]
#            print(fi[j])
#            print(sig[i])
    return A

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


###_________podatki___________________________________________ 



DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sesta/farmakoloski.dat'

###_________Fit_funkcije__________________________________________ 
    
def Lin_Fdata(x,k,n):#,c0,d0):  
    return k*x+n#+c0*x**0.9+abs(d0)*x**1.1 #c0*x**d0#+e0*np.log(f0*x)  
    
def Model_Fdata(x,y0,a):  
    return y0*x/(x+a)


###_________podatki___________________________________________ 


data = loadtxt(DIR) # branje
X=data[:,0]
Y=data[:,1]

sig_Y=np.ones(len(Y))*3

## linearizacija podatkov
v=1/X
u=1/Y
sig_u_p=abs(1/Y**2)*sig_Y

###_________MAIN reševanje___________________________________________ 

M=2

b=u/(abs(sig_Y/Y**2))          #vektor b izmerkov

A=gen_A(v,abs(sig_Y/Y**2),M)   #SVD razcep vektor b izmerkv
print(A)

U, S, Vh = np.linalg.svd(A, full_matrices=False)  #SVD razcep

a=gen_a(U, S, Vh ,b) ### vektor parametrov preko SVD razcepa
print('[n,k]='+str(a))
sig_a=np.sqrt(sig2(S,Vh)) ### vektor nrgotovosti parametrov preko SVD razcepa
print(r'[sig_n,sig_k]='+str(sig_a))

n,k=a

y0=1/n
a=k/n
chi=sum(((Y-Model_Fdata(X,y0,a))/sig_Y)**2)

## histogram odstopanj vrednosti modela od meritev

F10=plt.figure(10)
#plt.step(xfi,Hfi,'c',alpha = 0.5,label=r'odstopanja')
plt.step(Y-Model_Fdata(X,y0,a),'c',alpha = 0.5,label=r'odstopanja')
plt.xlabel(r'$ y_i-\widetilde{y}_i $')
plt.ylabel('n')
plt.legend(loc='best')

###_________poskusni___________________________________________ 
#