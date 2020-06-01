# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
import matplotlib.pyplot  as plt
from scipy import stats


e=2.718281828459045    
pi=3.141592653589793 

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

##___________________________ 

####     RAČUNANJE IN TABELIRANJE

DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/osma/podatki/' # pot do delovne mape s podatki
leto = ["TRandom1_1000000","TRandom2_1000000","TRandom3_1000000"]
naloga =["101","102","103","104","105","106","107","108","109","110","111","112"]
N=1000000
R1 = np.genfromtxt(DIR + leto[0]+".txt", dtype=None, comments="#", usecols=range(1)) # branje           
R2 = np.genfromtxt(DIR + leto[1]+".txt", dtype=None, comments="#", usecols=range(1)) # branje           
R3 = np.genfromtxt(DIR + leto[2]+".txt", dtype=None, comments="#", usecols=range(1)) # branje   



a1=np.random.rand(1000000)  # generiranje števil uniform

import cProfile
cProfile.run("np.random.rand(1000000)") #### 4 function calls in 0.028 seconds
####TRandom1_1000000 TStopwatch()  -> CpuTime()=0.33s
####TRandom2_1000000 TStopwatch()  -> CpuTime()=0.25s
####TRandom3_1000000 TStopwatch()  -> CpuTime()=0.24s

Ha1,xa1,binEdge_a1,La1=Hdata(a1)  # normaliziacija histograma za chi test 
HR1,xR1,binEdge_R1,LR1=Hdata(R1)  # normaliziacija histograma za chi test 
HR2,xR2,binEdge_R2,LR2=Hdata(R2)  # normaliziacija histograma za chi test 
HR3,xR3,binEdge_R3,LR3=Hdata(R3)  # normaliziacija histograma za chi test 
#
#X2a1=stats.chisquare(a1,axis=None)
#X2R1=stats.chisquare(R1,axis=None) # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
#X2R2=stats.chisquare(R2,axis=None) # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
#X2R3=stats.chisquare(R3,axis=None) # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev

X2a1=stats.chisquare(Ha1,axis=None)
X2R1=stats.chisquare(HR1,axis=None) # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
X2R2=stats.chisquare(HR2,axis=None) # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
X2R3=stats.chisquare(HR3,axis=None) # chi test za odtopanje generiranih uniform vrednosti glede na  uniform porazdelitev
#
KDa1=stats.kstest(a1, 'uniform')
KSR1=stats.kstest(R1, 'uniform')    # K-S test against normal function
KSR2=stats.kstest(R2, 'uniform')    # K-S test against normal function
KSR3=stats.kstest(R3, 'uniform')    # K-S test against normal function

#AAA=0.01
#AAA=0.5
#AAA=0.99
#np.sqrt(-0.5*np.log(AAA*0.5))/np.sqrt(100)

#KDa1=stats.kstest(Ha1, 'uniform')
#KSR1=stats.kstest(HR1, 'uniform')    # K-S test against normal function
#KSR2=stats.kstest(HR2, 'uniform')    # K-S test against normal function
#KSR3=stats.kstest(HR3, 'uniform')  


F2=plt.figure(1)
F2=plt.subplot(1, 2, 1 )
plt.hist(a1, bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{random.rand(N)}\ (\nu = $'+str(La1-1)+r', $\chi^2=$'+'{:.{}f}'.format(X2a1[0], 1 )+r', p='+'{:.{}f}'.format(X2a1[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(R1, bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{TRandom1}\ (\nu = $'+str(LR1-1)+r', $\chi^2=$'+'{:.{}f}'.format(X2R1[0], 1 )+r', p='+'{:.{}f}'.format(X2R1[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(R2, bins='auto',normed=True,facecolor="None",edgecolor='g',label=r'$\tt{TRandom2}\ (\nu = $'+str(LR2-1)+r', $\chi^2=$'+'{:.{}f}'.format(X2R2[0], 1 )+r', p='+'{:.{}f}'.format(X2R2[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(R3, bins='auto',normed=True,facecolor="None",edgecolor='b',label=r'$\tt{TRandom3}\ (\nu = $'+str(LR3-1)+r', $\chi^2=$'+'{:.{}f}'.format(X2R3[0], 1 )+r', p='+'{:.{}f}'.format(X2R3[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.title('Normirani histogrami enakomerne porazdelitve  '+'(N='+str(1000000)+')')
plt.xlabel('x')
plt.ylabel('n')
plt.legend(loc=4)
plt.show()
F2=plt.subplot(1,2, 2 )
plt.hist(a1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r'$\tt{random.rand(N)}\ (D=$'+'{:.{}f}'.format(KDa1[0], 3 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KDa1[0]*np.sqrt(N), 3 )+r', p='+'{:.{}f}'.format(KDa1[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(R1, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='r',label=r'$\tt{TRandom1}\ (D=$'+'{:.{}f}'.format(KSR1[0], 4 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KSR1[0]*np.sqrt(N), 2 )+r', p='+'{:.{}f}'.format(KSR1[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(R2, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='g',label=r'$\tt{TRandom2}\ (D=$'+'{:.{}f}'.format(KSR2[0], 4 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KSR2[0]*np.sqrt(N), 2 )+r', p='+'{:.{}f}'.format(KSR2[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(R3, bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='b',label=r'$\tt{TRandom3}\ (D=$'+'{:.{}f}'.format(KSR3[0], 4 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KSR3[0]*np.sqrt(N), 2 )+r', p='+'{:.{}f}'.format(KSR3[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.title('Kumilativni normirani histogrami enakomerne porazdelitve  '+'(N='+str(1000000)+')')
plt.ylabel('kumulativen(n)')
plt.xlabel('x')
plt.legend(loc=2)
plt.show()   

