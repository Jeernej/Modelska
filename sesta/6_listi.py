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


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sesta/CdL3_linfit.dat'

###_________Modeli__________________________________________ 
    

    
def chi2(y_m,y_t,sig_y_m):
    chi=sum(((y_m-y_t)/sig_y_m)**2)    
    return chi


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




def model(std,C_O,C_S):  
    std_S=std[0:len(std)/2]
    std_O=std[len(std)/2:len(std)]
    mod=C_O*std_O+C_S*std_S
    return mod 
    

     
###_________podatki______________________________________________________________________________________ 


data = loadtxt(DIR) # branje

energija=data[:,0]
epi=data[:,1]
mezo=data[:,2]
S_GHS=data[:,3]
O_pect=data[:,4]
S_GHS_O_pect=np.append(S_GHS,O_pect)

### računanje modelskih parametrov ______________________________________________________________________________________ 
#
err_result_Epi, err_result_Epi_cov=curve_fit(model,S_GHS_O_pect,epi)
print(err_result_Epi)
print(err_result_Epi_cov)
corr_result_Epi_cov = np.sqrt(np.diag(err_result_Epi_cov))
#chi2_epi=chi2(epi-model(S_GHS_O_pect,err_result_Epi[0],err_result_Epi[1]),sig_mezo)

err_result_Mezo, err_result_Mezo_cov=curve_fit(model,S_GHS_O_pect,mezo)
print(err_result_Mezo)
print(err_result_Mezo_cov)
corr_result_Mezo_cov = np.sqrt(np.diag(err_result_Mezo_cov))
#chi2_mezo=chi2(mezo-model(S_GHS_O_pect,err_result_Mezo[0],err_result_Mezo[1]),sig_mezo)


###_____ploti____________
#

F20=plt.figure(20)
plt.title('Absorpcijski spektri kadmija na robu	$ L_3 $',fontsize=16)
plt.plot(energija,epi,color='g',label=r'Cd-O,S (epi vzorec)',markersize=2)
plt.plot(energija,mezo,color='y',label=r'Cd-O,S (mezo vzorec)',markersize=2)
plt.plot(energija,S_GHS,  color='r',LS='--', label=r'Cd-O (GHS standard)',markersize=2)
plt.plot(energija,O_pect, color='b',LS='--', label=r'Cd-S (pektin standard)',markersize=2)
#plt.plot(x,Model_Fdata(x, 106.32, 24.76 ), color="#f78d6f",LS=':',label='basic fit; [$y_0$=106.32, $a$=24.76]' )
#plt.fill_between(x, Model_Fdata(x, 106.32+4.758940, 24.76+4.163827 ),Model_Fdata(x, 106.32-4.758940, 24.76-4.163827 ), color="#f78d6f",alpha=0.2,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )

#plt.plot(x,Model_Fdata(x,*err_result_Model),'b:', label=r'fit z $\sigma_y$; [$y_0$='+'{:.{}f}'.format(err_result_Model[0], 2)+r', $a$='+'{:.{}f}'.format(err_result_Model[1], 2)+']' )
#corr_result_Model_cov = np.sqrt(np.diag(err_result_Model_cov))
#plt.fill_between(x, Model_Fdata(x,*[err_result_Model[0]+corr_result_Model_cov[0],err_result_Model[1]+corr_result_Model_cov[1]]),Model_Fdata(x,*[err_result_Model[0]-corr_result_Model_cov[0],err_result_Model[1]-corr_result_Model_cov[1]]), color="#6fd9f7",alpha=0.2,label=r'fit z $\sigma_y$ negotovost; [$\sigma_{y_0}$='+'{:.{}f}'.format(corr_result_Model_cov[0], 2)+r', $\sigma_{a}$='+'{:.{}f}'.format(corr_result_Model_cov[1], 2)+']' )

plt.xlabel(r'$E$',fontsize=16)
plt.ylabel(r'$\frac{dP}{dE}$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=0)

F10=plt.figure(10)
F10=plt.subplot(2,2, 1)  
plt.title('Absorpcijski spektri kadmija na robu $L_3 $ za Cd-O,S (epi vzorec)',fontsize=18)
plt.plot(energija,epi,color='g',label=r'meritve',markersize=2)
plt.plot(energija,model(S_GHS_O_pect,err_result_Epi[0],err_result_Epi[1]),color='g',LS='--',label=r'model ; ($C_O=$'+'{:.{}f}'.format(err_result_Epi[0], 2)+r'$\pm$'+'{:.{}f}'.format(corr_result_Epi_cov[0], 2)+r',$C_S=$'+'{:.{}f}'.format(err_result_Epi[1], 2)+r'$\pm$'+'{:.{}f}'.format(corr_result_Epi_cov[1], 2)+')',markersize=2)
plt.xlabel(r'$E$',fontsize=20)
plt.ylabel(r'$\frac{dP}{dE}$',fontsize=20)
#plt.xlim([0,1100])
plt.legend(loc=0)

F10=plt.subplot(2,2, 3)  
plt.title('Absorpcijski spektri kadmija na robu $L_3 $ za Cd-O,S (mezo vzorec)',fontsize=18)
plt.plot(energija,mezo,color='y',label=r'meritve',markersize=2)
plt.plot(energija,model(S_GHS_O_pect,err_result_Mezo[0],err_result_Mezo[1]),color='y',LS='--',label=r'model ; ($C_O=$'+'{:.{}f}'.format(err_result_Mezo[0], 2)+r'$\pm$'+'{:.{}f}'.format(corr_result_Mezo_cov[0], 2)+r',$C_S=$'+'{:.{}f}'.format(err_result_Mezo[1], 2)+r'$\pm$'+'{:.{}f}'.format(corr_result_Mezo_cov[1], 2)+')',markersize=2)
plt.xlabel(r'$E$',fontsize=20)
plt.ylabel(r'$\frac{dP}{dE}$',fontsize=20)
#plt.xlim([0,1100])
plt.legend(loc=0)

Hepi, xepi, binEdge_epi, Lfi=Hdata(epi-model(S_GHS_O_pect,err_result_Epi[0],err_result_Epi[1]))
Hmezo, xmezo, binEdge_mezo, Lfi=Hdata(mezo-model(S_GHS_O_pect,err_result_Mezo[0],err_result_Mezo[1]))

F10=plt.subplot(2,2, 2)  
plt.title(r'Porazdelitev odstopanj med meritvami $y_i$ in modelom $\widetilde{y}_i$  (epi vzorec)',fontsize=18)
plt.step(xepi,Hepi,'g',alpha = 0.95,label=r'Cd-O,S (epi vzorec)')
#plt.step(Y-Model_Fdata(X,y0,a),'c',alpha = 0.5,label=r'odstopanja')
plt.xlabel(r'$ y_i-\widetilde{y}_i $',fontsize=20)
plt.ylabel('n',fontsize=20)
plt.legend(loc='best')

F10=plt.subplot(2,2, 4)  
plt.title(r'Porazdelitev odstopanj med meritvami $y_i$ in modelom $\widetilde{y}_i$  (mezo vzorec)',fontsize=18)
plt.step(xmezo,Hmezo,'y',alpha = 0.95,label=r'Cd-O,S (mezo vzorec)')
#plt.step(Y-Model_Fdata(X,y0,a),'c',alpha = 0.5,label=r'odstopanja')
plt.xlabel(r'$ y_i-\widetilde{y}_i $',fontsize=20)
plt.ylabel('n',fontsize=20)
plt.legend(loc='best')

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
