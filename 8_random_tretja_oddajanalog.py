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
    H, binEdge = np.histogram(dogodki, bins='auto',normed=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram, polažaji sredine binov, število binov

def Hdata10(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins=10,normed=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram, polažaji sredine binov, število binov
 
def Hdata6(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins=6,normed=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram, polažaji sredine binov, število binov
         
#____________________________________________________ 

#     RAČUNANJE IN TABELIRANJE

DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/osma/podatki/' # pot do delovne mape s podatki
leto = ["mod_tm10_","mod_tm11_","mod_tm13_","mod_tm14_"]
naloga =["101","102","103","104","105","106","107","108","109","110","111","112"]

l=0
LETO=[0,0,0,0]
while l<len(leto):
    LET=np.empty(0) 
    n=0
    while n<len(naloga):
#        print('leto',n,l)
        IN = np.genfromtxt(DIR + leto[l]+naloga[n]+".dat", dtype=None, comments="#", delimiter=":", usecols=range(3)) # branje           
        i=0
        while i<len(leto):
            casmin=IN[i][0]*60*24+IN[i][1]*60+IN[i][2]
            if casmin>-800:
                LET=np.append(LET,casmin)
            i=i+1
        n=n+1
    LETO[l]=LET
    l=l+1
    
n=0
NALOGA=[0,0,0,0,0,0,0,0,0,0,0,0]
while n<len(naloga) : 
    NAL=np.empty(0)       
    l=0
    while l<len(leto):
#        print('naloga',n,l)
        IN = np.genfromtxt(DIR + leto[l]+naloga[n]+".dat", dtype=None, comments="#", delimiter=":", usecols=range(3)) # branje   
        i=1
        while i<len(leto):
            casmin=IN[i][0]*60*24+IN[i][1]*60+IN[i][2]
            if casmin>-800:
                NAL=np.append(NAL,casmin)
            i=i+1      
        l=l+1
    NALOGA[n]=NAL
    n=n+1
    
HN1, xN1, binEdge_N1, LN1 =Hdata(NALOGA[0])
HN2, xN2, binEdge_N2, LN2 =Hdata(NALOGA[1])
HN3, xN3, binEdge_N3, LN3 =Hdata(NALOGA[2])
HN4, xN4, binEdge_N4, LN4 =Hdata(NALOGA[3])
HN5, xN5, binEdge_N5, LN5 =Hdata(NALOGA[4])
HN6, xN6, binEdge_N6, LN6 =Hdata(NALOGA[5])
HN7, xN7, binEdge_N7, LN7 =Hdata(NALOGA[6])
HN8, xN8, binEdge_N8, LN8 =Hdata(NALOGA[7])
HN9, xN9, binEdge_N9, LN9 =Hdata(NALOGA[8])
HN10, xN10, binEdge_N10, LN10 =Hdata(NALOGA[9])
HN11, xN11, binEdge_N11, LN11 =Hdata(NALOGA[10])
HN12, xN12, binEdge_N12, LN12 =Hdata(NALOGA[11])

HL10, x10, binEdge_10, L10=Hdata(LETO[0])  
HL11, x11, binEdge_11, L11=Hdata(LETO[1])
HL13, x13, binEdge_13, L13=Hdata(LETO[2])
HL14, x14, binEdge_14, L14=Hdata(LETO[3])

#HN1, xN1, binEdge_N1, LN1 =Hdata6(NALOGA[0])
#HN2, xN2, binEdge_N2, LN2 =Hdata6(NALOGA[1])
#HN3, xN3, binEdge_N3, LN3 =Hdata6(NALOGA[2])
#HN4, xN4, binEdge_N4, LN4 =Hdata6(NALOGA[3])
#HN5, xN5, binEdge_N5, LN5 =Hdata6(NALOGA[4])
#HN6, xN6, binEdge_N6, LN6 =Hdata6(NALOGA[5])
#HN7, xN7, binEdge_N7, LN7 =Hdata6(NALOGA[6])
#HN8, xN8, binEdge_N8, LN8 =Hdata6(NALOGA[7])
#HN9, xN9, binEdge_N9, LN9 =Hdata6(NALOGA[8])
#HN10, xN10, binEdge_N10, LN10 =Hdata6(NALOGA[9])
#HN11, xN11, binEdge_N11, LN11 =Hdata6(NALOGA[10])
#HN12, xN12, binEdge_N12, LN12 =Hdata6(NALOGA[11])
#
#HL10, x10, binEdge_10, L10=Hdata10(LETO[0])  
#HL11, x11, binEdge_11, L11=Hdata10(LETO[1])
#HL13, x13, binEdge_13, L13=Hdata10(LETO[2])
#HL14, x14, binEdge_14, L14=Hdata10(LETO[3])

KS10=stats.kstest(LETO[0], 'norm')    # K-S test against normal function
KS11=stats.kstest(LETO[1], 'norm')    # K-S test against normal function
KS13=stats.kstest(LETO[2], 'norm')    # K-S test against normal function
KS14=stats.kstest(LETO[3], 'norm')    # K-S test against normal function
    

#KS10=stats.kstest(HL10, 'norm')    # K-S test against normal function
#KS11=stats.kstest(HL11, 'norm')    # K-S test against normal function
#KS13=stats.kstest(HL13, 'norm')    # K-S test against normal function
#KS14=stats.kstest(HL14, 'norm')    # K-S test against normal function
    



crta=['k','c','m','y','b','r','g']

F2=plt.figure(1)
F2=plt.subplot(2, 1, 1 )
plt.hist(NALOGA[0], bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{nal\ 1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[1], bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{nal\ 2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[2], bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{nal\ 3}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[3], bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{nal\ 4}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[4], bins='auto',normed=True,facecolor="None",edgecolor='b',label=r'$\tt{nal\ 5}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[5], bins='auto',normed=True,facecolor="None",edgecolor='y',label=r'$\tt{nal\ 6}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[6], bins='auto',normed=True,facecolor="None",edgecolor='g',label=r'$\tt{nal\ 7}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[7], bins='auto',normed=True,facecolor="None",edgecolor='c',ls=':',label=r'$\tt{nal\ 8}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[8], bins='auto',normed=True,facecolor="None",edgecolor='m',ls=':',label=r'$\tt{nal\ 9}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[9], bins='auto',normed=True,facecolor="None",edgecolor='r',ls=':',label=r'$\tt{nal\ 10}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[10], bins='auto',normed=True,facecolor="None",edgecolor='k',ls=':',label=r'$\tt{nal\ 11}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[11], bins='auto',normed=True,facecolor="None",edgecolor='b',ls=':',label=r'$\tt{nal\ 12}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Normirani histogrami porazdelitev časov oddaje nalog za posamezne naloge skozi leta')#+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel('t[min]')
plt.ylabel('n')
plt.legend(loc=4)
plt.show()
F2=plt.subplot(2, 1, 2 )
plt.hist(NALOGA[0], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r'$\tt{nal\ 1}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[1], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',label=r'$\tt{nal\ 2}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[2], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='r',label=r'$\tt{nal\ 3}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[3], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='k',label=r'$\tt{nal\ 4}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[4], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='b',label=r'$\tt{nal\ 5}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[5], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='y',label=r'$\tt{nal\ 6}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[6], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='g',label=r'$\tt{nal\ 7}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[7], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',ls=':',label=r'$\tt{nal\ 8}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[8], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',ls=':',label=r'$\tt{nal\ 9}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[9], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='r',ls=':',label=r'$\tt{nal\ 10}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[10], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='k',ls=':',label=r'$\tt{nal\ 11}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(NALOGA[11], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='b',ls=':',label=r'$\tt{nal\ 12}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Kumilativno za posamezne naloge skozi leta')#+'(N='+str(N)+',M='+str(M)+')')
plt.ylabel('kumulativen(n)')
plt.xlabel('t[min]')
plt.legend(loc=4)
plt.show()

F3=plt.figure(2)
F3=plt.subplot(2, 1, 1 )
plt.hist(LETO[0], bins='auto',normed=True,facecolor="None",edgecolor='c',label=r'$\tt{leto\ 10/11}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(LETO[1], bins='auto',normed=True,facecolor="None",edgecolor='m',label=r'$\tt{leto\ 11/12}$',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(LETO[2], bins='auto',normed=True,facecolor="None",edgecolor='r',label=r'$\tt{leto\ 13/14}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(LETO[3], bins='auto',normed=True,facecolor="None",edgecolor='k',label=r'$\tt{leto\ 14/15}$',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Normirani histogrami porazdelitev časov oddaje nalog za posamezna leta')#+'(N='+str(N)+',M='+str(M)+')')
plt.xlabel('t[min]')
plt.ylabel('n')
plt.legend(loc=4)
plt.show()
F3=plt.subplot(2, 1, 2 )
plt.hist(LETO[0], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='c',label=r"$\tt{kstest(leto\ 10/11, 'norm')}=(D=$"+'{:.{}f}'.format(KS10[0], 3 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KS10[0]*np.sqrt(len(LETO[0])), 2 )+r',p='+r', p='+'{:.{}f}'.format(KS10[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(LETO[1], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='m',label=r"$\tt{kstest(leto\ 11/12, 'norm')}=(D=$"+'{:.{}f}'.format(KS11[0], 3 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KS11[0]*np.sqrt(len(LETO[1])), 2 )+r',p='+r', p='+'{:.{}f}'.format(KS11[1], 3 )+')',alpha = 0.5,histtype='step')  # arguments are passed to np.histogram
plt.hist(LETO[2], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='r',label=r"$\tt{kstest(leto\ 13/14, 'norm')}=(D=$"+'{:.{}f}'.format(KS13[0], 3 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KS13[0]*np.sqrt(len(LETO[2])), 2 )+r',p='+r', p='+'{:.{}f}'.format(KS13[1], 3 )+')',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.hist(LETO[3], bins='auto',normed=True,cumulative=True,facecolor="None",edgecolor='k',label=r"$\tt{kstest(leto\ 14/15, 'norm')}=(D=$"+'{:.{}f}'.format(KS14[0], 3 )+r' ,$D\sqrt{N}=$'+'{:.{}f}'.format(KS14[0]*np.sqrt(len(LETO[3])), 2 )+r',p='+r', p='+'{:.{}f}'.format(KS14[1], 3 )+')',alpha = 0.4,histtype='step')  # arguments are passed to np.histogram
plt.title('Kumilativno za posamezna leta')#+'(N='+str(N)+',M='+str(M)+')')
plt.ylabel('kumulativen(n)')
plt.xlabel('t[min]')
plt.legend(loc=4)
plt.show()
#plt.suptitle('Normirani histogrami porazdelitev časov oddaje nalog',fontsize=16)
