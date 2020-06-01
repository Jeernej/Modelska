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

#
#font = {
#        'color':  'k',
##        'weight': 'bold',
#        'size': 26,
#        'verticalalignment': 'bottom'
#        }
#font_AX = {
#        'color':  'k',
#        'size': 30,
#        }
        
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
     
     
###_________Modeli__________________________________________ 
    

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


###_________Fit_funkcije__________________________________________ 

    
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
    
def Lin_Fdata(x,k,n):#,c0,d0):  #funkcija za fitanje gaussove porazdelitve
#    return a0+b0*x+c0*x**0.5+d0*x**0.3+g0*x**h0 #c0*x**d0#+e0*np.log(f0*x)   
    return k*x+n#+c0*x**0.9+abs(d0)*x**1.1 #c0*x**d0#+e0*np.log(f0*x)  
    
def Model_Fdata(x,y0,a):  #funkcija za fitanje gaussove porazdelitve
    return y0*x/(x+a)


###_________podatki___________________________________________ 


data = loadtxt(DIR) # branje
X=data[:,0]
Y=data[:,1]

sig_Y=np.ones(len(Y))*3

v=1/X
u=1/Y
sig_u_p=abs(1/Y**2)*sig_Y
sig_u_m=abs(1/Y**2)*sig_Y
#sig_u_p=abs(1/Y - 1/(Y+sig_Y) )
#sig_u_m=abs(1/Y - 1/(Y-sig_Y) )
#sig_u_m=(1/Y)/(1-(sig_Y/Y))
sig_u=[abs(sig_u_m), abs(sig_u_p)]


#
#data = loadtxt(DIR) # branje
#FI_fp=data[:,0]
#for i in range(0,len(FI_fp)) : FI_fp[i]=math.radians(FI_fp[i])
#X_fp=data[:,1]
#
#TH_tg=data[:,2]
#for i in range(0,len(TH_tg)) : TH_tg[i]=math.radians(TH_tg[i])
#
#sig_FI_fp=np.ones(len(FI_fp))/1000
#sig_X_fp=np.ones(len(X_fp))
#sig_TH_tg=np.ones(len(TH_tg))/1000


M=2

b=u/(abs(sig_Y/Y**2))
A=gen_A(v,abs(sig_Y/Y**2),M)
print(A)
#A[:,0]=1
U, S, Vh = np.linalg.svd(A, full_matrices=False)

a=gen_a(U, S, Vh ,b)
sig_a=np.sqrt(sig2(S,Vh))
print(a)
print(sig_a)

n,k=a

y0=1/n
a=k/n
chi=sum(((Y-Model_Fdata(X,y0,a))/sig_Y)**2)


Hfi, xfi, binEdge_fi, Lfi=Hdata(Y-Model_Fdata(X,y0,a)  )

F10=plt.figure(10)
plt.step(xfi,Hfi,'c',alpha = 0.5,label=r'odstopanja')
#plt.step(Y-Model_Fdata(X,y0,a),'c',alpha = 0.5,label=r'odstopanja')
plt.xlabel(r'$ y_i-\widetilde{y}_i $')
plt.ylabel('n')
plt.legend(loc='best')

###_________poskusni___________________________________________ 
#
#
#gmodel_Lin = Model(Lin_Fdata)
#gmodel_Model = Model(Model_Fdata)
##gmodel_Lin.param_names
##gmodel_Lin.independent_vars
##Y_Lin = gmodel_Lin.eval(x=X, a0=0.01 , b0=0.001 , c0=5. , d0=0.05 )
##result_Lin = gmodel_Lin.fit(data=y,x=x, a0=66. , b0=0.3 , c0=-1.64 , d0=6.0, method='leastsq' )
#result_Lin = gmodel_Lin.fit(data=u, x=v, k=0.2 , n=0.01 ,method='leastsq' )
#print(result_Lin.fit_report())
#
#result_Model = gmodel_Model.fit(data=Y, x=X, y0=100. , a=21 ,method='leastsq' )
#print(result_Model.fit_report())
#
##
#err_result_Lin, err_result_Lin_cov=curve_fit(Lin_Fdata,fp,tg, sigma=sig_u_p)
#print(err_result_Lin)
#print(err_result_Lin_cov)
#
#
#err_result_Model, err_result_Model_cov=curve_fit(Model_Fdata,X,Y, sigma=sig_Y)
#print(err_result_Model)
#print(err_result_Model_cov)
#
###  ploting corrected FIT


###_____ploti____________
##
#x=np.linspace(1,len(FI_fp),len(FI_fp))
#
#F10=plt.figure(3)
##F10=plt.subplot(1,2, 1 )  
#plt.title('Kalibracijski podatki visokoločljivostnega magnetnega spektrometra',fontsize=16)
#plt.errorbar(x,FI_fp, yerr=sig_FI_fp, color='r', marker='o',label=r'$ \vartheta_{tg} $ [rad]',markersize=2)
#plt.errorbar(x,X_tg, yerr=sig_X_tg, color='g', marker='o',label=r'$ x_{fp} $ [mm]',markersize=2)
#plt.errorbar(x,TH_tg, yerr=sig_TH_tg, color='b', marker='o',label=r'$ \vartheta_{fp}$ [rad]',markersize=2)
##plt.plot(x,Model_Fdata(x, 106.32, 24.76 ), color="#f78d6f",LS=':',label='basic fit; [$y_0$=106.32, $a$=24.76]' )
##plt.fill_between(x, Model_Fdata(x, 106.32+4.758940, 24.76+4.163827 ),Model_Fdata(x, 106.32-4.758940, 24.76-4.163827 ), color="#f78d6f",alpha=0.2,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
#
##plt.plot(x,Model_Fdata(x,*err_result_Model),'b:', label=r'fit z $\sigma_y$; [$y_0$='+'{:.{}f}'.format(err_result_Model[0], 2)+r', $a$='+'{:.{}f}'.format(err_result_Model[1], 2)+']' )
##corr_result_Model_cov = np.sqrt(np.diag(err_result_Model_cov))
##plt.fill_between(x, Model_Fdata(x,*[err_result_Model[0]+corr_result_Model_cov[0],err_result_Model[1]+corr_result_Model_cov[1]]),Model_Fdata(x,*[err_result_Model[0]-corr_result_Model_cov[0],err_result_Model[1]-corr_result_Model_cov[1]]), color="#6fd9f7",alpha=0.2,label=r'fit z $\sigma_y$ negotovost; [$\sigma_{y_0}$='+'{:.{}f}'.format(corr_result_Model_cov[0], 2)+r', $\sigma_{a}$='+'{:.{}f}'.format(corr_result_Model_cov[1], 2)+']' )
#
#plt.xlabel(r'$št. meritve$',fontsize=16)
##plt.ylabel(r'$y(x)$',fontsize=16)
##plt.xlim([0,1100])
#plt.legend(loc=0)
#
#
#xx=np.linspace(0,max(v))
#
#F10=plt.subplot(1,2, 2 )  
#plt.title('Linearizirani merjeni farmakološki podatki odziva na dozo',fontsize=16)
#plt.errorbar(v,u, yerr=sig_u, color='k',marker='o', label='meritve',markersize=2)
#
#plt.plot(xx,Lin_Fdata(xx, 9180, -705 ), color='r',LS=':',label='fit; [$1/n$=-'+'{:.{}f}'.format(1/705,4)+', $k/n$='+'{:.{}f}'.format(9180/705,2)+']' )
#plt.fill_between(xx, Lin_Fdata(xx, 9180+1.54e+03, -705+615.7412 ),Lin_Fdata(xx, 9180-1.54e+03, -705-615.7412 ), color="#f78d6f",alpha=0.2,label=r'fit negotovost; [$\sigma_{1/n}$='+'{:.{}f}'.format(615.7/705.6**2,3)+', $\sigma_{k/n}$='+'{:.{}f}'.format(9180/705.6 * np.sqrt((615.7/705.6)**2+(1540/9180)**2),3)+']' )
#
#plt.plot(xx,Lin_Fdata(xx,*err_result_Lin),'b:', label=r'fit z $\sigma_u$; [$1/n$='+'{:.{}f}'.format(1/err_result_Lin[1], 3)+', $k/n$='+'{:.{}f}'.format(err_result_Lin[0]/err_result_Lin[1], 3)+']' )
#corr_result_Lin_cov = np.sqrt(np.diag(err_result_Lin_cov))
#plt.fill_between(xx, Lin_Fdata(xx,*[err_result_Lin[0]+corr_result_Lin_cov[0],err_result_Lin[1]+corr_result_Lin_cov[1]]),Lin_Fdata(xx,*[err_result_Lin[0]-corr_result_Lin_cov[0],err_result_Lin[1]-corr_result_Lin_cov[1]]), color="#6fd9f7",alpha=0.2,label=r'fit z $\sigma_u$ negotovost; [$\sigma_{1/n}$='+'{:.{}f}'.format(corr_result_Lin_cov[1]/err_result_Lin[1]**2, 2)+', $\sigma_{k/n}$='+'{:.{}f}'.format((err_result_Lin[0]/err_result_Lin[1])*np.sqrt((corr_result_Lin_cov[1]/err_result_Lin[1])**2+(corr_result_Lin_cov[0]/err_result_Lin[0])**2), 2)+']' )
#
#plt.xlabel(r'$v$',fontsize=16)
#plt.ylabel(r'$u(v)$',fontsize=16)
#plt.xlim([0,1.2])
#plt.ylim([10**(-4),10**9])
#plt.yscale('log')
#plt.legend(loc='best')


##plt.plot(x, result_Lin.init_fit, 'b--', label='init_fit')
#plt.plot(x, result_Lin.best_fit, 'r-', label='Fit')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',c=+'+'{:.{}f}'.format(SHfi, 3 )+r',d=+'+'{:.{}f}'.format(SHfi, 3 )+')')
#dely1_LIN = result_Lin.eval_uncertainty(sigma=1)
#dely2_LIN = result_Lin.eval_uncertainty(sigma=2)
#dely3_LIN = result_Lin.eval_uncertainty(sigma=3)
#plt.fill_between(x, result_Lin.best_fit-dely3_LIN, result_Lin.best_fit+dely3_LIN, color="#cccccc",label=r'confidence $\sigma=3$')
#plt.fill_between(x, result_Lin.best_fit-dely2_LIN, result_Lin.best_fit+dely2_LIN, color="#ABABAB",label=r'confidence $\sigma=2$')
#plt.fill_between(x, result_Lin.best_fit-dely1_LIN, result_Lin.best_fit+dely1_LIN, color="#888888",label=r'confidence $\sigma=1$')
#


##F10=plt.subplot(3, 1, 2 )  
#F10=plt.subplot(1,2, 2 )  
#plt.plot(x, y, 'ko', label='data')

##plt.plot(x, result_Log.init_fit, 'b--', label='init_fit')
#plt.plot(x, result_Log.best_fit, 'r-', label='Log_fit  ')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',e=+'+'{:.{}f}'.format(SHfi, 3 )+r',f=+'+'{:.{}f}'.format(SHfi, 3 )+')')
#dely1_LOG = result_Log.eval_uncertainty(sigma=1)
#dely2_LOG = result_Log.eval_uncertainty(sigma=2)
#dely3_LOG = result_Log.eval_uncertainty(sigma=3)
#plt.fill_between(x, result_Log.best_fit-dely3_LOG, result_Log.best_fit+dely3_LOG, color="#cccccc",label=r'confidence $\sigma=3$')
#plt.fill_between(x, result_Log.best_fit-dely2_LOG, result_Log.best_fit+dely2_LOG, color="#ABABAB",label=r'confidence $\sigma=2$')
#plt.fill_between(x, result_Log.best_fit-dely1_LOG, result_Log.best_fit+dely1_LOG, color="#888888",label=r'confidence $\sigma=1$')

###____________poskusni__podatki______________________________________ 

#
#plt.plot(S1x1, S1y1, color="y",alpha=0.6)# color="#cccccc")
#plt.plot(S1x2, S1y2, color="y",alpha=0.6)#  color="#cccccc")
##plt.plot(S2x1, S2y1, color="#f57a57")
##plt.plot(S2x2, S2y2, color="#f57a57")
#
#plt.plot(X,Lin_fdata(X), 'k-', label=r'Fit extrapolation $\rightarrow$   $Teol\approx '+str(x2[0])+'h$')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',c=+'+'{:.{}f}'.format(SHfi, 3 )+r',d=+'+'{:.{}f}'.format(SHfi, 3 )+')')
#dely1_LIN = result_Lin.eval_uncertainty(sigma=1)
#dely2_LIN = result_Lin.eval_uncertainty(sigma=2)
###dely3_LIN = result_Lin.eval_uncertainty(sigma=4)
###plt.fill_between(xL, result_Lin.best_fit-dely3_LIN, result_Lin.best_fit+dely3_LIN, color="#cccccc",label=r'confidence $3\sigma$')
#plt.fill_between(X, Lin_fdataM(X),Lin_fdatam(X), color="y",alpha=0.4,label=r'$\pm\ fit\ param.\  error$ $\rightarrow$   $Teol^{+'+str(S1x1[0]-x2[0])+'}_{-'+str(x2[0]-S1x2[0])+ '}\ [h]$')#  $\rightarrow$   $Teol^{+'+str(S1x1[0]-x2[0])+'}_{-'+str(x2[0]-S1x2[0])+ '}$ [$h$]')
#plt.fill_between(xL, result_Lin.best_fit-dely2_LIN, result_Lin.best_fit+dely2_LIN, color="#f78d6f",label=r'confidence $\pm 2\sigma$ ')# $\rightarrow$   $Teol^{+'+str(S2x1[0]-x2[0])+'}_{-'+str(x2[0]-S2x2[0])+ '}$ [$h$]')
#plt.fill_between(xL, result_Lin.best_fit-dely1_LIN, result_Lin.best_fit+dely1_LIN, color="#6fd8f7",label=r'confidence $\pm\sigma$ ')#  $\rightarrow$   $Teol^{+'+str(S1x1[0]-x2[0])+'}_{-'+str(x2[0]-S1x2[0])+ '}$ [$h$]')
##plt.plot(X, Lin_fdata(X), 'b-', label='model fit')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',c=+'+'{:.{}f}'.format(SHfi, 3 )+r',d=+'+'{:.{}f}'.format(SHfi, 3 )+')')
##plt.plot(X, Lin_fdatam(X), 'r:', label='Fit funkcija')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',c=+'+'{:.{}f}'.format(SHfi, 3 )+r',d=+'+'{:.{}f}'.format(SHfi, 3 )+')')
##plt.plot(X, Lin_fdataM(X), 'r--', label='Fit funkcija')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',c=+'+'{:.{}f}'.format(SHfi, 3 )+r',d=+'+'{:.{}f}'.format(SHfi, 3 )+')')
#
#
#plt.plot(X, C, 'k:')#, label='Lin_fit')#+r'(a=+'+'{:.{}f}'.format(SHfi, 3 )+r',b=+'+'{:.{}f}'.format(SHfi, 3 )+r',c=+'+'{:.{}f}'.format(SHfi, 3 )+r',d=+'+'{:.{}f}'.format(SHfi, 3 )+')')
#plt.plot(X, S, ls=':',color='#cccccc')
#plt.plot(x2, y2, 'k:')
#plt.xlabel(r'$time\ [h]$', fontdict=font_AX)
##plt.ylabel(r'IR [$\mu \Omega$]', fontdict=font)
#plt.ylabel(r'$ESR/ESR_{rated}\ [\%]$', fontdict=font_AX)
#plt.xlim([-50,DOSEG])
#plt.minorticks_on()
##plt.title('Data for 500F $2^{nd}$ batch test:   '+podatki[0] +'\n with linear model fit:  $a_0+b_0 \cdot x$  '+' ($a_0='+'{:.{}f}'.format(a0, 2 )+'\pm'+'{:.{}f}'.format(A0, 2 )+', b_0='+'{:.{}f}'.format(b0, 4 )+'\pm'+'{:.{}f}'.format(B0, 4 )+'$)',fontdict=font)
#plt.title('500F $2^{nd}$ batch: '+Podatki[0] +'\n linear fit: $a_0+b_0 \cdot x$ '+'\n ($a_0='+'{:.{}f}'.format(a0, 2 )+'\pm'+'{:.{}f}'.format(A0, 2 )+', b_0='+'{:.{}f}'.format(b0, 4 )+'\pm'+'{:.{}f}'.format(B0, 4 )+'$)',fontdict=font)
#plt.legend(loc=2,fontsize=18)
#plt.tick_params(axis='both', which='major', labelsize=26)
