# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
#import scipy as sc
import matplotlib.pyplot  as plt
#from scipy import stats
#from scipy.optimize import curve_fit
from lmfit import Model
from numpy import loadtxt
from scipy.optimize import curve_fit


font = {
        'color':  'k',
#        'weight': 'bold',
        'size': 26,
        'verticalalignment': 'bottom'
        }
font_AX = {
        'color':  'k',
        'size': 30,
        }
        
e=2.718281828459045    
pi=3.141592653589793 


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
##sig_u_m=(1/Y)/(1-(sig_Y/Y))
#sig_u=[abs(sig_u_m), abs(sig_u_p)]

###_________poskusni___________________________________________ 


gmodel_Lin = Model(Lin_Fdata)
gmodel_Model = Model(Model_Fdata)
#gmodel_Lin.param_names
#gmodel_Lin.independent_vars
#Y_Lin = gmodel_Lin.eval(x=X, a0=0.01 , b0=0.001 , c0=5. , d0=0.05 )
#result_Lin = gmodel_Lin.fit(data=y,x=x, a0=66. , b0=0.3 , c0=-1.64 , d0=6.0, method='leastsq' )
result_Lin = gmodel_Lin.fit(data=u, x=v, k=0.2 , n=0.01 ,method='leastsq' )
print(result_Lin.fit_report())

result_Model = gmodel_Model.fit(data=Y, x=X, y0=100. , a=21 ,method='leastsq' )
print(result_Model.fit_report())

#
err_result_Lin, err_result_Lin_cov=curve_fit(Lin_Fdata,v,u, sigma=sig_u_p)
print(err_result_Lin)
print(err_result_Lin_cov)


err_result_Model, err_result_Model_cov=curve_fit(Model_Fdata,X,Y, sigma=sig_Y)
print(err_result_Model)
print(err_result_Model_cov)

##  ploting corrected FIT


k,n=err_result_Lin
y0=1/n
a=k/n
sig_n,sig_k= np.sqrt(np.diag(err_result_Lin_cov))


sig_y0=y0*(sig_n/n)
sig_a=a*np.sqrt((sig_n/n)**2 +(sig_k/k)**2)

chi=sum(((Y-Model_Fdata(X,y0,a))/sig_Y)**2)

#chi=sum(((u-Lin_Fdata(v,k,n))/sig_u_p)**2)


###_____ploti____________
#
x=np.linspace(0,max(X))

F10=plt.figure(3)
F10=plt.subplot(1,2, 1 )  
plt.title('Merjeni farmakološki podatki odziva na dozo',fontsize=16)
plt.errorbar(X,Y, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
#plt.plot(x,Model_Fdata(x, 106.32, 24.76 ), color="#f78d6f",LS=':',label='basic fit; [$y_0$=106.32, $a$=24.76]' )
#plt.fill_between(x, Model_Fdata(x, 106.32+4.758940, 24.76+4.163827 ),Model_Fdata(x, 106.32-4.758940, 24.76-4.163827 ), color="#f78d6f",alpha=0.2,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )

plt.plot(x,Model_Fdata(x,*err_result_Model),'b:', label=r'fit with $\sigma$; [$y_0$='+'{:.{}f}'.format(err_result_Model[0], 2)+r', $a$='+'{:.{}f}'.format(err_result_Model[1], 2)+']' )
corr_result_Model_cov = np.sqrt(np.diag(err_result_Model_cov))
plt.fill_between(x, Model_Fdata(x,*[err_result_Model[0]+corr_result_Model_cov[0],err_result_Model[1]+corr_result_Model_cov[1]]),Model_Fdata(x,*[err_result_Model[0]-corr_result_Model_cov[0],err_result_Model[1]-corr_result_Model_cov[1]]), color="#6fd9f7",alpha=0.2,label=r'fit with $\sigma$ error; [$\sigma_{y_0}$='+'{:.{}f}'.format(corr_result_Model_cov[0], 2)+r', $\sigma_{a}$='+'{:.{}f}'.format(corr_result_Model_cov[1], 2)+']' )


plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1100])
plt.legend(loc=4)


xx=np.linspace(0,max(v))

F10=plt.subplot(1,2, 2 )  
plt.title('Linearizirani merjeni farmakološki podatki odziva na dozo',fontsize=16)
plt.errorbar(v,u, yerr=sig_u, color='k',marker='o', label='meritve',markersize=2)

plt.plot(xx,Lin_Fdata(xx, 9180, -705 ), color='r',LS=':',label='basic fit; [$k$=9180, $n$=-705]' )
plt.fill_between(xx, Lin_Fdata(xx, 9180+1.54e+03, -705+615.7412 ),Lin_Fdata(xx, 9180-1.54e+03, -705-615.7412 ), color="#f78d6f",alpha=0.2,label=r'basic fit error; [$\sigma_{k}$=1540, $\sigma_{n}$=615.7]' )

plt.plot(xx,Lin_Fdata(xx,*err_result_Lin),'b:', label=r'fit with $\sigma$; [$k$='+'{:.{}f}'.format(err_result_Lin[0], 3)+r', $n$='+'{:.{}f}'.format(err_result_Lin[1], 3)+']' )
corr_result_Lin_cov = np.sqrt(np.diag(err_result_Lin_cov))
plt.fill_between(xx, Lin_Fdata(xx,*[err_result_Lin[0]+corr_result_Lin_cov[0],err_result_Lin[1]+corr_result_Lin_cov[1]]),Lin_Fdata(xx,*[err_result_Lin[0]-corr_result_Lin_cov[0],err_result_Lin[1]-corr_result_Lin_cov[1]]), color="#6fd9f7",alpha=0.2,label=r'fit with $\sigma$ error; [$\sigma_{k}$='+'{:.{}f}'.format(corr_result_Lin_cov[0], 3)+r', $\sigma_{n}$='+'{:.{}f}'.format(corr_result_Lin_cov[1], 5)+']' )

plt.xlabel(r'$v$',fontsize=16)
plt.ylabel(r'$u(v)$',fontsize=16)
plt.xlim([0,1.2])
#plt.ylim([-5000,25000])
plt.yscale('log')
plt.legend(loc='best')


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
