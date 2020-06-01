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


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sedma/farmakoloski.dat'

###_________Fit_funkcije__________________________________________ 

def Model_Fdata(x,y0,a):  #funkcija za fitanje gaussove porazdelitve
    return y0*x/(x+a)

def P_Model_Fdata(x,y0,a,p):  #funkcija za fitanje gaussove porazdelitve
    return y0*x**p/(x**p+a**p)

###_________podatki___________________________________________ 


data = loadtxt(DIR) # branje
X=data[:,0]
Y=data[:,1]

sig_Y=np.ones(len(Y))*3


###_________poskus
#

err_result_Model, err_result_Model_cov=curve_fit(Model_Fdata,X,Y, sigma=sig_Y,method='lm')
print(err_result_Model)
print(err_result_Model_cov)

err_result_PModel, err_result_PModel_cov=curve_fit(P_Model_Fdata,X,Y, sigma=sig_Y, method='lm')
print(err_result_PModel)
print(err_result_PModel_cov)

##  ploting corrected FIT


###_____ploti____________
#
x=np.linspace(0,max(X))

F10=plt.figure(3)
#F10=plt.subplot(1,2, 1 )  
plt.title('Merjeni farmakološki podatki odziva na dozo',fontsize=16)
plt.errorbar(X,Y, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
#plt.plot(x,Model_Fdata(x, 106.32, 24.76 ), color="#f78d6f",LS=':',label='basic fit; [$y_0$=106.32, $a$=24.76]' )
#plt.fill_between(x, Model_Fdata(x, 106.32+4.758940, 24.76+4.163827 ),Model_Fdata(x, 106.32-4.758940, 24.76-4.163827 ), color="#f78d6f",alpha=0.2,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )

y_m=Model_Fdata(X,*[err_result_Model[0],err_result_Model[1]])
chi=sum(((Y-y_m)/sig_Y)**2)/(len(X)-2)

plt.plot(x,Model_Fdata(x,*err_result_Model),'b:', label=r'fit z $\sigma_y$; [$y_0$='+'{:.{}f}'.format(err_result_Model[0], 2)+r', $a$='+'{:.{}f}'.format(err_result_Model[1], 2)+']   ; $\chi^2=$'+'{:.{}f}'.format(chi, 2) )
corr_result_Model_cov = np.sqrt(np.diag(err_result_Model_cov))
plt.fill_between(x, Model_Fdata(x,*[err_result_Model[0]+corr_result_Model_cov[0],err_result_Model[1]+corr_result_Model_cov[1]]),Model_Fdata(x,*[err_result_Model[0]-corr_result_Model_cov[0],err_result_Model[1]-corr_result_Model_cov[1]]), color="#6fd9f7",alpha=0.2,label=r'fit z $\sigma_y$ negotovost; [$\sigma_{y_0}$='+'{:.{}f}'.format(corr_result_Model_cov[0], 2)+r', $\sigma_{a}$='+'{:.{}f}'.format(corr_result_Model_cov[1], 2)+']' )

yp_m=P_Model_Fdata(X,*[err_result_PModel[0],err_result_PModel[1],err_result_PModel[2]])
P_chi=sum(((Y-yp_m)/sig_Y)**2)/(len(X)-3)

plt.plot(x,P_Model_Fdata(x,*err_result_PModel),'r:', label=r'fit z $\sigma_y$; [$y_0$='+'{:.{}f}'.format(err_result_PModel[0], 2)+r', $a$='+'{:.{}f}'.format(err_result_PModel[1], 2)+r', $p$='+'{:.{}f}'.format(err_result_PModel[2], 2)+']  ; $\chi^2=$'+'{:.{}f}'.format(P_chi, 2) )
corr_result_PModel_cov = np.sqrt(np.diag(err_result_PModel_cov))
plt.fill_between(x, P_Model_Fdata(x,*[err_result_PModel[0]+corr_result_PModel_cov[0],err_result_PModel[1]+corr_result_PModel_cov[1],err_result_PModel[2]+corr_result_PModel_cov[2]]),P_Model_Fdata(x,*[err_result_PModel[0]-corr_result_PModel_cov[0],err_result_PModel[1]-corr_result_PModel_cov[1],err_result_PModel[2]+corr_result_PModel_cov[2]]), color="#f78d6f",alpha=0.2,label=r'fit z $\sigma_y$ negotovost; [$\sigma_{y_0}$='+'{:.{}f}'.format(corr_result_PModel_cov[0], 2)+r', $\sigma_{a}$='+'{:.{}f}'.format(corr_result_PModel_cov[1], 2)+r', $\sigma_{p}$='+'{:.{}f}'.format(corr_result_PModel_cov[2], 2)+']' )

plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$y(x)$',fontsize=16)
plt.xlim([0,1100])
plt.legend(loc=4)

