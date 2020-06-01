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


DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/sedma/ledvice.dat'

###_________Fit_funkcije__________________________________________ 

### enorazdelčni
def Model(x,A,a): 
    return A*e**(-a*x)

def Q_Model(x,D,d):  
    return D*e**(-d*np.sqrt(x))

### enorazdelčni linearizirani
def LModel(x,A,a):  
    return -a*x+np.log(A)

def Q_LModel(x,A,a):  
    return -a*np.sqrt(x)+np.log(A)
    
    
### enorazdelčni + konstanta
def CModel(x,A,a,D):  
    return D+A*e**(-a*x)

def CQ_Model(x,A,a,D): 
    return D+A*e**(-a*np.sqrt(x))


    
### dvorazdelčni
def Model2(x,A,a,B,b):
    return A*e**(-a*x)+B*e**(-b*x)

def QModel2(x,A,a,B,b):  
    return A*e**(-a*np.sqrt(x))+B*e**(-b*np.sqrt(x))
 
    

    
### dvorazdelčni + konst
def C_Model2(x,A,a,B,b,C):
    return C+A*e**(-a*x)+B*e**(-b*x)

def CQ_Model2(x,A,B,a,b,C):
    return C+A*e**(-a*np.sqrt(x))+B*e**(-b*np.sqrt(x))


    
### dvorazdelčni + konst + korenski cas
def CQ_Model3(x,A,a,B,b,D,d,C): 
    return C+A*e**(-a*x)+B*e**(-b*x)+D*e**(-d*np.sqrt(x))

###_________podatki___________________________________________ 


data = loadtxt(DIR) # branje
cas=data[:,0]
sunki=data[:,1]

sig_Y=np.sqrt(sunki)  ## negotovost sunkov določimo s poissonovim kriterijem sig=sqrt(N)
sig_lY=sig_Y/sunki ## negotovost sunkov za linearizirano obliko


###_________izračuni___________________________________________ 

### enorazdelčni
resM, resM_cov=curve_fit(Model,cas,sunki, p0=[9000,0.0001],sigma=sig_Y, method='lm')
Sig_resM = np.sqrt(np.diag(resM_cov))
y_M=Model(cas,*[resM[0],resM[1]])
y_Mp=Model(cas,*[resM[0]+Sig_resM[0],resM[1]-Sig_resM[1]])
y_Mm=Model(cas,*[resM[0]-Sig_resM[0],resM[1]+Sig_resM[1]])
chi2_M=sum(((sunki-y_M)/sig_Y)**2)/(len(cas)-2)

resQM, resQM_cov=curve_fit(Q_Model,cas,sunki,sigma=sig_Y, method='lm')
Sig_resQM = np.sqrt(np.diag(resQM_cov))
y_QM=Q_Model(cas,*[resQM[0],resQM[1]])
y_QMp=Q_Model(cas,*[resQM[0]+Sig_resQM[0],resQM[1]-Sig_resQM[1]])
y_QMm=Q_Model(cas,*[resQM[0]-Sig_resQM[0],resQM[1]+Sig_resQM[1]])
chi2_QM=sum(((sunki-y_QM)/sig_Y)**2)/(len(cas)-2)

### enorazdelčni linearizirani
resLM, resLM_cov=curve_fit(LModel,cas,np.log(sunki),sigma=sig_lY, method='lm')
Sig_resLM = np.sqrt(np.diag(resLM_cov))
y_LM=LModel(cas,*[resLM[0],resLM[1]])
y_LMp=LModel(cas,*[resLM[0]+Sig_resLM[0],resLM[1]-Sig_resLM[1]])
y_LMm=LModel(cas,*[resLM[0]-Sig_resLM[0],resLM[1]+Sig_resLM[1]])
chi2_LM=sum(((np.log(sunki)-y_LM)/sig_lY)**2)/(len(cas)-2)

resLQM, resLQM_cov=curve_fit(Q_LModel,cas,np.log(sunki),sigma=sig_lY, method='lm')
Sig_resLQM = np.sqrt(np.diag(resLQM_cov))
y_LQM=Q_LModel(cas,*[resLQM[0],resLQM[1]])
y_LQMp=Q_LModel(cas,*[resLQM[0]+Sig_resLQM[0],resLQM[1]-Sig_resLQM[1]])
y_LQMm=Q_LModel(cas,*[resLQM[0]-Sig_resLQM[0],resLQM[1]+Sig_resLQM[1]])
chi2_LQM=sum(((np.log(sunki)-y_LQM)/sig_lY)**2)/(len(cas)-2)


### enorazdelčni (korenski) + konst

resCM, resCM_cov=curve_fit(CModel,cas,sunki, p0=[9000,0.001,100],sigma=sig_Y, method='lm')
Sig_resCM = np.sqrt(np.diag(resCM_cov))
y_CM=CModel(cas,*[resCM[0],resCM[1],resCM[2]])
y_CMp=CModel(cas,*[resCM[0]+Sig_resCM[0],resCM[1]-Sig_resCM[1],resCM[2]+Sig_resCM[2]])
y_CMm=CModel(cas,*[resCM[0]-Sig_resCM[0],resCM[1]+Sig_resCM[1],resCM[2]-Sig_resCM[2]])
chi2_CM=sum(((sunki-y_CM)/sig_Y)**2)/(len(cas)-3)

resCQM, resCQM_cov=curve_fit(CQ_Model,cas,sunki, p0=[0.01,1,11000], sigma=sig_Y, method='lm')
Sig_resCQM = np.sqrt(np.diag(resCQM_cov))
y_CQM=CQ_Model(cas,*[resCQM[0],resCQM[1],resCQM[2]])
y_CQMp=CQ_Model(cas,*[resCQM[0]+Sig_resCQM[0],resCQM[1]-Sig_resCQM[1],resCQM[2]+Sig_resCQM[2]])
y_CQMm=CQ_Model(cas,*[resCQM[0]-Sig_resCQM[0],resCQM[1]+Sig_resCQM[1],resCQM[2]-Sig_resCQM[2]])
chi2_CQM=sum(((sunki-y_CQM)/sig_Y)**2)/(len(cas)-3)


### dvorazdelčni (korenski) 

resM2, resM2_cov=curve_fit(Model2,cas,sunki,p0=[6000,0.007,6000,0.00001],sigma=sig_Y, method='lm')
Sig_resM2 = np.sqrt(np.diag(resM2_cov))
y_M2=Model2(cas,*[resM2[0],resM2[1],resM2[2]],resM2[3])
y_M2p=Model2(cas,*[resM2[0]+Sig_resM2[0],resM2[1]-Sig_resM2[1],resM2[2]+Sig_resM2[2],resM2[3]-Sig_resM2[3]])
y_M2m=Model2(cas,*[resM2[0]-Sig_resM2[0],resM2[1]+Sig_resM2[1],resM2[2]-Sig_resM2[2],resM2[3]+Sig_resM2[3]])
chi2_M2=sum(((sunki-y_M2)/sig_Y)**2)/(len(cas)-4)

resQM2, resQM2_cov=curve_fit(QModel2,cas,sunki,p0=[6000,0.007,6000,0.00001],sigma=sig_Y, method='lm')
Sig_resQM2 = np.sqrt(np.diag(resQM2_cov))
y_QM2=QModel2(cas,*[resQM2[0],resQM2[1],resQM2[2]],resQM2[3])
y_QM2p=QModel2(cas,*[resQM2[0]+Sig_resQM2[0],resQM2[1]-Sig_resQM2[1],resQM2[2]+Sig_resQM2[2],resQM2[3]-Sig_resQM2[3]])
y_QM2m=QModel2(cas,*[resQM2[0]-Sig_resQM2[0],resQM2[1]+Sig_resQM2[1],resQM2[2]-Sig_resQM2[2],resQM2[3]+Sig_resQM2[3]])
chi2_QM2=sum(((sunki-y_QM2)/sig_Y)**2)/(len(cas)-4)


### dvorazdelčni (korenski) + konst
resCM2, resCM2_cov=curve_fit(C_Model2,cas,sunki,p0=[6000,0.007,6000,0.001,2000],sigma=sig_Y, method='lm')
Sig_resCM2 = np.sqrt(np.diag(resCM2_cov))
y_CM2=C_Model2(cas,*[resCM2[0],resCM2[1],resCM2[2]],resCM2[3],resCM2[4])
y_CM2p=C_Model2(cas,*[resCM2[0]+Sig_resCM2[0],resCM2[1]-Sig_resCM2[1],resCM2[2]+Sig_resCM2[2],resCM2[3]-Sig_resCM2[3],resCM2[4]+Sig_resCM2[4]])
y_CM2m=C_Model2(cas,*[resCM2[0]-Sig_resCM2[0],resCM2[1]+Sig_resCM2[1],resCM2[2]-Sig_resCM2[2],resCM2[3]+Sig_resCM2[3],resCM2[4]-Sig_resCM2[4]])
chi2_CM2=sum(((sunki-y_CM2)/sig_Y)**2)/(len(cas)-5)

resCQM2, resCQM2_cov=curve_fit(CQ_Model2,cas,sunki,p0=[10000,0.07,10000,0.01,200],sigma=sig_Y, method='lm')
Sig_resCQM2 = np.sqrt(np.diag(resCQM2_cov))
y_CQM2=CQ_Model2(cas,*[resCQM2[0],resCQM2[1],resCQM2[2]],resCQM2[3],resCQM2[4])
y_CQM2p=CQ_Model2(cas,*[resCQM2[0]+Sig_resCQM2[0],resCQM2[1]-Sig_resCQM2[1],resCQM2[2]+Sig_resCQM2[2],resCQM2[3]-Sig_resCQM2[3],resCQM2[4]+Sig_resCQM2[4]])
y_CQM2m=CQ_Model2(cas,*[resCQM2[0]-Sig_resCQM2[0],resCQM2[1]+Sig_resCQM2[1],resCQM2[2]-Sig_resCQM2[2],resCQM2[3]+Sig_resCQM2[3],resCQM2[4]-Sig_resCQM2[4]])
chi2_CQM2=sum(((sunki-y_CQM2)/sig_Y)**2)/(len(cas)-5)


### dvorazdelčni + konst + korenski 

resCQM3, resCQM3_cov=curve_fit(CQ_Model3,cas,sunki,p0=[6000,0.007,6000,0.001,1000,0.001,2000],sigma=sig_Y, method='lm')
Sig_resCQM3 = np.sqrt(np.diag(resCQM3_cov))
y_CQM3=CQ_Model3(cas,*[resCQM3[0],resCQM3[1],resCQM3[2]],resCQM3[3],resCQM3[4],resCQM3[5],resCQM3[6])
y_CQM3p=CQ_Model3(cas,*[resCQM3[0]+Sig_resCQM3[0],resCQM3[1]-Sig_resCQM3[1],resCQM3[2]+Sig_resCQM3[2],resCQM3[3]-Sig_resCQM3[3],resCQM3[4]+Sig_resCQM3[4],resCQM3[5]-Sig_resCQM3[5],resCQM3[6]+Sig_resCQM3[6]])
y_CQM3m=CQ_Model3(cas,*[resCQM3[0]-Sig_resCQM3[0],resCQM3[1]+Sig_resCQM3[1],resCQM3[2]-Sig_resCQM3[2],resCQM3[3]+Sig_resCQM3[3],resCQM3[4]-Sig_resCQM3[4],resCQM3[5]+Sig_resCQM3[5],resCQM3[6]-Sig_resCQM3[6]])
chi2_CQM3=sum(((sunki-y_CQM3)/sig_Y)**2)/(len(cas)-6)


######_____ploti________############____

x=np.linspace(0,max(cas))

###################### enorazdelčni (korenski) linearizirani

F10=plt.figure(3)
F10=plt.subplot(1,2, 1 )  
plt.title('Merjeni klinični podatki za čistost ledvic',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(cas,sunki, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
plt.plot(cas,y_M, color="b",LS='-',label=r'$Ae^{-at}$ ; [$A$='+'{:.{}f}'.format(resM[0],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resM[0], 2)+r',$a$='+'{:.{}e}'.format(resM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM[1], 1)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_M, 1) )
plt.fill_between(cas,y_Mp,y_Mm, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
plt.plot(cas,y_QM, color="r",LS='-',label=r'$Ae^{-a\sqrt{t}}$ ; [$A$='+'{:.{}f}'.format(resQM[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resQM[0], 0)+r',$a$='+'{:.{}e}'.format(resQM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resQM[1], 1)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_QM, 1) )
plt.fill_between(cas,y_QMp,y_QMm, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$N$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=1)

F10=plt.subplot(1,2, 2 )  
plt.title('Linearizirani merjeni klinični podatki za čistost ledvic',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(cas,np.log(sunki), yerr=sig_lY, color='k', marker='o',label='meritve',markersize=2)
plt.plot(cas,y_LM, color="b",LS='-',label=r'$-at +ln(A)$ ; [$A$='+'{:.{}f}'.format(resLM[0],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resLM[0], 0)+r',$a$='+'{:.{}e}'.format(resLM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resLM[1], 1)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_LM, 1) )
plt.fill_between(cas,y_LMp,y_LMm, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
plt.plot(cas,y_LQM, color="r",LS='-',label=r'$-a\sqrt{t} +ln(A)$ ; [$A$='+'{:.{}f}'.format(resLQM[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resLQM[0], 0)+r',$a$='+'{:.{}e}'.format(resLQM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resLQM[1], 1)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_LQM, 1) )
plt.fill_between(cas,y_LQMp,y_LQMm, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$\ln{N}$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=1)


###################### enorazdelčni (korenski) + konst
F11=plt.figure(11)
plt.title('Merjeni klinični podatki za čistost ledvic',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(cas,sunki, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
plt.plot(cas,y_CM, color="b",LS='-',label=r'$Ae^{-at}+D$ ; [$A$='+'{:.{}f}'.format(resCM[0],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCM[0], 0)+r',$a$='+'{:.{}e}'.format(resCM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM[1], 1)+r',$D$='+'{:.{}f}'.format(resCM[2], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCM[2], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CM, 1) )
plt.fill_between(cas,y_CMp,y_CMm, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
plt.plot(cas,y_CQM, color="r",LS='-',label=r'$Ae^{-a\sqrt{t}}+D$ ; [$A$='+'{:.{}f}'.format(resCQM[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM[0], 0)+r',$a$='+'{:.{}e}'.format(resCQM[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCQM[1],1)+r',$D$='+'{:.{}f}'.format(resCQM[2], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM[2], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CQM, 1) )
plt.fill_between(cas,y_CQMp,y_CQMm, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$N$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=1)


######################### dvorazdelčni (korenski) + konst

F20=plt.figure(2)
F20=plt.subplot(2,1, 1 )  
plt.title('Merjeni klinični podatki za čistost ledvic',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(cas,sunki, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
plt.plot(cas,y_M2, color="b",LS='-',label=r'$Ae^{-at}+BAe^{-bt}$ ; [$A$='+'{:.{}f}'.format(resM2[0],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resM2[0], 0)+r',$a$='+'{:.{}e}'.format(resM2[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[1], 1)+',$B$='+'{:.{}f}'.format(resM2[2],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resM2[2], 0)+r',$b$='+'{:.{}e}'.format(resM2[3],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resM2[3], 1)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_M2, 1) )
plt.fill_between(cas,y_M2p,y_M2m, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
plt.plot(cas,y_QM2, color="r",LS='-',label=r'$Ae^{-a\sqrt{t}}+Be^{-b\sqrt{t}}$ ; [$A$='+'{:.{}f}'.format(resQM2[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resQM2[0], 0)+r',$a$='+'{:.{}e}'.format(resQM2[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resQM2[1], 1)+',$B$='+'{:.{}f}'.format(resQM2[2],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resQM2[2], 0)+r',$b$='+'{:.{}e}'.format(resQM2[3], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resQM2[3], 1)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_QM2, 1) )
#plt.fill_between(cas,y_QM2p,y_QM2m, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$N$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=1)


F20=plt.subplot(2,1, 2 )  
plt.title('Merjeni klinični podatki za čistost ledvic',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(cas,sunki, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
plt.plot(cas,y_CM2, color="b",LS='-',label=r'$Ae^{-at}+BAe^{-bt}+D$ ; [$A$='+'{:.{}f}'.format(resCM2[0],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCM2[0], 0)+r',$a$='+'{:.{}e}'.format(resCM2[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM2[1],1)+',$B$='+'{:.{}f}'.format(resCM2[2],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCM2[2], 0)+r',$b$='+'{:.{}e}'.format(resCM2[3], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCM2[3], 1)+r',$D$='+'{:.{}f}'.format(resCM2[4], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCM2[4], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CM2, 1) )
plt.fill_between(cas,y_CM2p,y_CM2m, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )
plt.plot(cas,y_CQM2, color="r",LS='-',label=r'$Ae^{-a\sqrt{t}}+Be^{-b\sqrt{t}}+D$ ; [$A$='+'{:.{}f}'.format(resCQM2[0], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM2[0], 0)+r',$a$='+'{:.{}f}'.format(resCQM2[1], 3)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM2[1], 3)+',$B$='+'{:.{}f}'.format(resCQM2[2],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM2[2], 0)+r',$b$='+'{:.{}f}'.format(resCQM2[3], 2)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM2[3], 2)+r',$D$='+'{:.{}f}'.format(resCQM2[4], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM2[4], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CQM2, 1) )
plt.fill_between(cas,y_CQM2p,y_CQM2m, color="#f78d6f",alpha=0.2)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$N$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=1)

################### OVERKILL ###### dvorazdelčni + korenski + konst

F30=plt.figure(30)
plt.title('Merjeni klinični podatki za čistost ledvic',fontsize=16)
#plt.plot(cas,sunki,  color='k', marker='o',label='meritve',markersize=2)
plt.errorbar(cas,sunki, yerr=sig_Y, color='k', marker='o',label='meritve',markersize=2)
plt.plot(cas,y_CQM3, color="b",LS='-',label=r'$Ae^{-at}+BAe^{-bt}+Ce^{-c\sqrt{t}}+D$ ; [$A$='+'{:.{}f}'.format(resCQM3[0],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM3[0], 0)+r',$a$='+'{:.{}e}'.format(resCQM3[1], 1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCQM3[1], 1)+',$B$='+'{:.{}f}'.format(resCQM3[2],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM3[2], 0)+r',$b$='+'{:.{}e}'.format(resCQM3[3],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCQM3[3], 1)+',$C$='+'{:.{}f}'.format(resCQM3[4],0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM3[4], 0)+r',$c$='+'{:.{}e}'.format(resCQM3[5],1)+r'$\pm$'+'{:.{}e}'.format(Sig_resCQM3[5], 1)+r',$D$='+'{:.{}f}'.format(resCQM3[6], 0)+r'$\pm$'+'{:.{}f}'.format(Sig_resCQM3[6], 0)+r'] ; $\chi^2=$'+'{:.{}f}'.format(chi2_CQM3, 1) )
#plt.fill_between(cas,y_CQM3p,y_CQM3m, color="#6fd9f7",alpha=0.2)#,label=r'basic fit error; [$\sigma_{y_0}$=4.76, $\sigma_{a}$=4.16]' )

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$N$',fontsize=16)
#plt.xlim([0,1100])
plt.legend(loc=1)