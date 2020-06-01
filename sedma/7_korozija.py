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
#from lmfit import Model
from numpy import loadtxt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

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


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sedma/korozija.dat'

###_________Fit_funkcije__________________________________________ 

### enorazdelčni + konstanta
def CModel(U,I0,Ua,Uc):  
    return I0*(e**(U/Ua)-e**(U/Uc)) 

def jacobian(U,I0,Ua,Uc): 
    dK     = e**(U/Ua)-e**(U/Uc)
    dzeta  = - e**(U/Ua) * I0 * U / Ua**2
    domegan= - e**(U/Uc) * I0 * U / Uc**2
    return np.transpose([dK,dzeta,domegan])
    
def LCModel(U,A,B,C):  
    return A*U+B*U**2+C*U**3
    
### dvorazdelčni
def Model2(U,I0,Ua,Uc,U0):
    return I0*(e**( (U-U0)/Ua)-e**( (U-U0)/Uc)) 

###_________podatki___________________________________________ 


data = loadtxt(DIR) # branje
napetost=data[:,0]
tok=data[:,1]

sig_Y=10**(-3)  ## negotovost sunkov ocenim
#sig_Y=0

###_________izračuni___________________________________________ 


#resCM, resCM_cov=curve_fit(CModel,napetost,tok,p0=[0.002,130, 70],sigma=sig_Y,method='lm')
resCM, resCM_cov=curve_fit(CModel,napetost,tok,p0=[0.01,100,100],sigma=sig_Y,method='lm')
#resCM, resCM_cov=curve_fit(CModel,napetost,tok,p0=[0.01,100,100],method='lm')
Sig_resCM = np.sqrt(np.diag(resCM_cov))
y_CM=CModel(napetost,*[resCM[0],resCM[1],resCM[2]])
y_CMp=CModel(napetost,*[resCM[0]+Sig_resCM[0],resCM[1]-Sig_resCM[1],resCM[2]-Sig_resCM[2]])
y_CMm=CModel(napetost,*[resCM[0]-Sig_resCM[0],resCM[1]+Sig_resCM[1],resCM[2]+Sig_resCM[2]])
chi2_CM=sum(((tok-y_CM)/sig_Y)**2)/(len(napetost)-3)
#chi2_CM=sum(((tok-y_CM))**2)/(len(napetost)-3)


#resCMS, resCMS_cov=curve_fit(CModel,napetost,tok,p0=[0.001,100,100],sigma=sig_Y,method='lm')
#Sig_resCMS = np.sqrt(np.diag(resCMS_cov))
#y_CMS=CModel(napetost,*[resCMS[0],resCMS[1],resCMS[2]])
#y_CMSp=CModel(napetost,*[resCMS[0]+Sig_resCMS[0],resCMS[1]-Sig_resCMS[1],resCMS[2]-Sig_resCMS[2]])
#y_CMSm=CModel(napetost,*[resCMS[0]-Sig_resCMS[0],resCMS[1]+Sig_resCMS[1],resCMS[2]+Sig_resCMS[2]])
#chi2_CMS=sum(((tok-y_CMS))**2)/(len(napetost)-3)



#### dvorazdelčni (korenski) 
#
#resM2, resM2_cov=curve_fit(Model2,napetost,tok,p0=[0.003,200,70,-5],sigma=sig_Y, method='lm')
resM2, resM2_cov=curve_fit(Model2,napetost,tok,p0=[0.01,100,100,-5],sigma=sig_Y, method='lm')
#resM2, resM2_cov=curve_fit(Model2,napetost,tok,p0=[0.01,100,100,-5], method='lm')
Sig_resM2 = np.sqrt(np.diag(resM2_cov))
y_M2=Model2(napetost,*[resM2[0],resM2[1],resM2[2]],resM2[3])
y_M2p=Model2(napetost,*[resM2[0]+Sig_resM2[0],resM2[1]-Sig_resM2[1],resM2[2]+Sig_resM2[2],resM2[3]-Sig_resM2[3]])
y_M2m=Model2(napetost,*[resM2[0]-Sig_resM2[0],resM2[1]+Sig_resM2[1],resM2[2]-Sig_resM2[2],resM2[3]+Sig_resM2[3]])
chi2_M2=sum(((tok-y_M2)/sig_Y)**2)/(len(napetost)-4)
#chi2_M2=sum(((tok-y_M2))**2)/(len(napetost)-4)

#resQM2, resQM2_cov=curve_fit(QModel2,cas,sunki,p0=[6000,0.007,6000,0.00001],sigma=sig_Y, method='lm')
#Sig_resQM2 = np.sqrt(np.diag(resQM2_cov))
#y_QM2=QModel2(cas,*[resQM2[0],resQM2[1],resQM2[2]],resQM2[3])
#y_QM2p=QModel2(cas,*[resQM2[0]+Sig_resQM2[0],resQM2[1]-Sig_resQM2[1],resQM2[2]+Sig_resQM2[2],resQM2[3]-Sig_resQM2[3]])
#y_QM2m=QModel2(cas,*[resQM2[0]-Sig_resQM2[0],resQM2[1]+Sig_resQM2[1],resQM2[2]-Sig_resQM2[2],resQM2[3]+Sig_resQM2[3]])
#chi2_QM2=sum(((sunki-y_QM2)/sig_Y)**2)/(len(cas)-4)


######_____ploti________############____


###################### enorazdelčni (korenski) + konst
F11=plt.figure(11)
plt.title('Meritev korozije med kovino in elektrolitom',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(napetost,tok, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
#plt.plot(napetost,y_CM, color="b",LS='-',label=r' eksaktno ; [$I_0$='+'{:.{}e}'.format(resCM[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[0],1)+r', $U_a$='+'{:.{}e}'.format(resCM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[1], 1)+r', $U_c$='+'{:.{}e}'.format(resCM[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[2], 1)+r'] ; $\sigma_I^2\chi^2=$'+'{:.{}e}'.format(chi2_CM, 1) )
#plt.plot(napetost,y_CM, color="r",LS='-',label=r' $\sigma_I=10^{-4}$ ; [$I_0$='+'{:.{}e}'.format(resCM[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[0],1)+r', $U_a$='+'{:.{}e}'.format(resCM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[1], 1)+r', $U_c$='+'{:.{}e}'.format(resCM[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[2], 1)+r'] ; $\chi^2=$'+'{:.{}e}'.format(chi2_CM, 1) )
plt.plot(napetost,y_CM, color="g",LS='-',label=r' $\sigma_I=10^{-3}$ ; [$I_0$='+'{:.{}e}'.format(resCM[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[0],1)+r', $U_a$='+'{:.{}e}'.format(resCM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[1], 1)+r', $U_c$='+'{:.{}e}'.format(resCM[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[2], 1)+r'] ; $\chi^2=$'+'{:.{}e}'.format(chi2_CM, 1) )
#plt.fill_between(napetost,y_CMp,y_CMm, color="#6fd9f7",alpha=0.2)
#plt.plot(napetost,y_CMS, color="r",LS='-',label=r'$ \sigma_I=10^{-4}$  ; [$I_0$='+'{:.{}e}'.format(resCMS[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCMS[0], 1)+r',$U_a$='+'{:.{}e}'.format(resCMS[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCMS[1], 1)+r',$U_c$='+'{:.{}e}'.format(resCMS[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCMS[2], 1)+r'] ; $\sigma_I^2\chi^2=$'+'{:.{}e}'.format(chi2_CMS, 1) )
#plt.fill_between(napetost,y_CMSp,y_CMSm, color="#f78d6f",alpha=0.2)
#plt.plot(napetost,y_CQM, color="r",LS='-',label=r'$Ae^{-a\sqrt{t}}+D$ ; [$A$='+'{:.{}f}'.format(resCQM[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM[0], 0)+r',$a$='+'{:.{}e}'.format(resCQM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCQM[1],1)+r',$D$='+'{:.{}f}'.format(resCQM[2], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM[2], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CQM, 1) )
#plt.fill_between(napetost,y_CQMp,y_CQMm, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$U$',fontsize=16)
plt.ylabel(r'$I$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=2)


F12=plt.figure(12)
plt.title('Meritev korozije med kovino in elektrolitom',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(napetost,tok, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
#plt.plot(napetost,y_M2, color="b",LS='-',label=r' eksaktno ; [$I_0$='+'{:.{}e}'.format(resM2[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[0],1)+r', $U_a$='+'{:.{}e}'.format(resM2[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[1], 1)+r', $U_c$='+'{:.{}e}'.format(resM2[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[2], 1)+r', $U_0$='+'{:.{}e}'.format(resM2[3], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[3], 1)+r'] ; $\sigma_I^2\chi^2=$'+'{:.{}e}'.format(chi2_M2, 1) )
#plt.plot(napetost,y_M2, color="r",LS='-',label=r' $\sigma_I=10^{-4}$ ; [$I_0$='+'{:.{}e}'.format(resM2[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[0],1)+r', $U_a$='+'{:.{}e}'.format(resM2[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[1], 1)+r', $U_c$='+'{:.{}e}'.format(resM2[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[2], 1)+r', $U_0$='+'{:.{}e}'.format(resM2[3], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[3], 1)+r'] ; $\chi^2=$'+'{:.{}e}'.format(chi2_M2, 1) )
plt.plot(napetost,y_M2, color="g",LS='-',label=r' $\sigma_I=10^{-3}$ ; [$I_0$='+'{:.{}e}'.format(resM2[0],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[0],1)+r', $U_a$='+'{:.{}e}'.format(resM2[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[1], 1)+r', $U_c$='+'{:.{}e}'.format(resM2[2], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[2], 1)+r', $U_0$='+'{:.{}e}'.format(resM2[3], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[3], 1)+r'] ; $\chi^2=$'+'{:.{}e}'.format(chi2_M2, 1) )
#plt.fill_between(napetost,y_CMp,y_CMm, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
#plt.plot(napetost,y_CQM, color="r",LS='-',label=r'$Ae^{-a\sqrt{t}}+D$ ; [$A$='+'{:.{}f}'.format(resCQM[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM[0], 0)+r',$a$='+'{:.{}e}'.format(resCQM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCQM[1],1)+r',$D$='+'{:.{}f}'.format(resCQM[2], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM[2], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CQM, 1) )
#plt.fill_between(napetost,y_CQMp,y_CQMm, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$U$',fontsize=16)
plt.ylabel(r'$I$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=2)
