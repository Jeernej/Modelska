# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:30:47 2019

@author: jernej


"""
import scipy as sc
from scipy import special
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import colors

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation


font = {
        'color':  'k',
        'size': 18,
        'verticalalignment': 'bottom'
        }
            
crta=['c','m','y','b','r','g','k']       

e=2.718281828459045    
pi=3.141592653589793 
e0=8.8541878128*10**(-12)


def Hdata(dogodki): # vse dogodke popredalčka in normalizira
    H, binEdge = np.histogram(dogodki, bins='auto',normed=True)
    
    L=len(binEdge)
    sred=(binEdge[1]-binEdge[0])/2.
    x=np.empty(L-1)
    k=0
    while k<L-1 :
        x[k]=binEdge[k]+sred
        k=k+1
    return  H,x,binEdge,L-1 #prebinan histogram (vrednost Hi,položajxi), polažaji sredine binov, število binov
     
     
def energija(zacetni,N): 
#    N=7   ## <-- !! popravi za vsak N
    en=0
    for i in range(0,N-1):
#        print('i='+str(i))
        for j in range(i+1,N):
#            en = en + np.sqrt( 2 - 2 * ( np.cos(zacetni[i])*np.cos(zacetni[j])  +  np.sin(zacetni[j])*np.sin(zacetni[i]) * np.sin(zacetni[i+N]) * np.sin(zacetni[j+N]) + np.sin(zacetni[j])*np.sin(zacetni[i]) * np.cos(zacetni[i+N]) * np.cos(zacetni[j+N]) ))**(-1)
            en = en + np.sqrt( 2 - 2 * ( np.cos(zacetni[i])*np.cos(zacetni[j])  +  np.sin(zacetni[j]) * np.sin(zacetni[i]) * np.sin(zacetni[i+N]) * np.sin(zacetni[j+N]) + np.sin(zacetni[j])*np.sin(zacetni[i]) * np.cos(zacetni[i+N]) * np.cos(zacetni[j+N]) ))**(-1)
#            print('j='+str(j))
#            print('en='+str(en))
    return en
    
    
def Izrac_tezisca(ZacetniAnim): 
    N=ZacetniAnim.shape[1]/2
    k=ZacetniAnim.shape[0]
    TEZ_X=np.empty(0)
    TEZ_Y=np.empty(0)    
    TEZ_Z=np.empty(0)    
    for i in range(0,k):
        polozaj=ZacetniAnim[i]   
        TH=polozaj[0:N]
        FI=polozaj[N:2*N]    
        xN=np.sin(TH)*np.cos(FI)
        yN=np.sin(TH)*np.sin(FI)
        zN=np.cos(TH) 
        TEZ_X=np.append(TEZ_X, sum(xN)/N)
        TEZ_Y=np.append(TEZ_Y, sum(yN)/N) 
        TEZ_Z=np.append(TEZ_Z, sum(zN)/N)
    return TEZ_X,TEZ_Y,TEZ_Z
    
    
    
def Izrac_momenta(ZacetniAnim,l,m): 
    
    N=ZacetniAnim.shape[1]/2
    k=ZacetniAnim.shape[0]

    MOM=np.empty(0)    

    for i in range(0,k):
        polozaj=ZacetniAnim[i]
    
        Th=polozaj[0:N]
        Fi=polozaj[N:2*N]
        MOM=np.append( MOM, sum(  np.real( sc.special.sph_harm(m,l, Fi, Th) ) )/ N )
    return MOM
    
      
def Izrac_povprečja(vektor): 
    AVG=np.empty(0)    
    for i in range(1,len(vektor)):
        AVG=np.append( AVG, np.mean(vektor[0:i]) )
    return AVG  
    
############################################################################################
    
NN=np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,40])
wiki=np.array([0.500000000,1.732050808,3.674234614,6.474691495,9.985281374,14.452977414,19.675287861,25.759986531,32.716949460,40.596450510,49.165253058,58.853230612,69.306363297,80.670244114,92.911655302,106.050404829,120.084467447,135.089467557,150.881568334,660.675278835])


N2=np.array([ -9.10779694e-08,   3.14159250e+00,   1.22289953e+00, 3.95281740e-01])
N3=np.array([ 1.90999692,  4.0928918 ,  0.41933584, -0.06874624,  0.375686  , 2.0380778 ])
N4=np.array([ 1.5153213 ,  1.8382184 ,  2.3220614 , -0.47044355,  6.02959814, 4.1215039 ,  1.73411481,  5.46672883])
N5=np.array([ 1.74682935,  1.70052893,  1.39474507,  0.42768616,  2.46637546, 0.11607488,  4.80524618,  3.25766707,  1.28596757,  1.91088074])
N6=np.array([ 1.57078009e+00,  -6.66270213e-06,   1.57082217e+00, 3.14159747e+00,   1.57080553e+00,   1.57079885e+00, 7.94168153e-01,   3.38829313e+00,   3.93577473e+00,  1.13495510e+00,   5.50657243e+00,   2.36499301e+00])
N7=np.array([ 3.00546654,  1.70433388,  1.45192809, -0.13610307,  1.59697856, 1.63027714,  1.47044451,  2.59768843, -0.41558198,  2.0928224 , 2.60133092,  0.84179427,  4.60611157,  3.35812576])
N8=np.array([ 1.60185338,  1.80285812,  2.96424905,  2.06229906,  0.80437882, 1.76252674,  1.24318638,  0.46788803,  1.17556254,  2.42531515, 0.52548832,  3.92333613,  3.25446006,  6.05693287,  4.90191975, 0.46539051])
N9=np.array([ 2.28433098,  1.14220957,  1.994061  ,  1.56390107,  2.65430292, 0.87015734,  1.91365831,  0.47666571,  1.23521502,  1.33863753, 3.67304198, -0.27008892,  4.84597724,  4.1567529 ,  2.06581988, 2.7183474 ,  5.55193448,  0.69440674])
N10=np.array([ 0.18596002,  1.57704488,  2.25039047,  2.58506499,  1.26011826, 1.93291597,  3.96018708,  1.20859586,  0.98889171,  1.23561421, 0.60607397,  0.57960662,  3.91431436,  1.30061388,  4.55372753, 2.62214458,  2.39164629, -0.51951284,  3.28168005,  1.70812455])
N11=np.array([ 1.69263524,  0.3347053 ,  1.33690829,  3.31212813,  1.93603419, 2.27789861,  1.7915341 ,  0.86407471,  0.91665268,  2.04103556, 1.20956935,  3.97096061,  3.45410113,  2.99804743,  2.10971662, 5.00768707,  2.51912041,  6.18305577,  6.84009218,  5.22923758, 1.06133235,  1.81920893])
N12=np.array([ 0.74758717,  0.58851543,  2.39404751,  1.73121614,  1.69165549, 1.2872355 ,  1.85435381,  1.41034905, -0.63088506,  3.73017722, 1.44996424,  2.5107371 ,  4.91056763,  0.64395768,  1.76905866, 2.82562195,  0.75876491,  1.80054242,  4.94222324, -0.31590032, -0.17165698,  0.64397968,  3.90034562, -0.17167691])
N13=np.array([ 1.70838824,  1.35767116,  0.69661723,  1.20054816,  2.29976904, 0.39711242,  1.28927136,  2.41016939,  1.83447009,  0.77887323, 2.06056307,  2.75463019,  1.62647514,  3.54348138,  1.39935087, 0.05614121,  2.71640513,  5.7020754 ,  2.0238356 ,  5.39812635, 1.04149077,  4.55428637,  4.20974669,  2.27908676,  3.66218857, 6.60522009])
N14=np.array([ 2.03757416,  2.03351482,  2.05002709,  3.15155045,  1.10749547, 1.09007897,  0.01003278,  1.09654909,  2.0533197 ,  2.03983755, 2.04709929,  1.10549463,  1.09984807,  1.08869737,  5.91629786, 0.67554607,  2.76681777,  3.69371252,  3.29627611,  1.20325792, 3.62424658,  5.38382449,  3.81897993,  1.7186918 ,  4.86997468, 4.33853409,  2.25233993,  0.15080814])
N15=np.array([ 2.02719112,  0.87389287,  3.36615187,  1.2311721 ,  1.33612788, 2.21047337,  1.43131233,  2.17737296,  1.3734411 ,  2.00559328, 1.02422606,  0.55861321,  1.7364989 ,  0.39071771,  2.29995353, 1.75464267,  2.04379619,  5.92813084,  5.34575295,  4.32623214, 4.6398853 ,  0.0537321 ,  5.71624936,  1.08470012,  3.54173606, 3.26546129,  0.44316869,  2.61055157,  4.44906674,  6.91337405])
N16=np.array([ 1.49635423,  2.10011244,  0.76517501,  1.61577394,  0.77047996, 1.57855934,  2.7907657 ,  1.03578353,  2.50328866,  1.73947495, 1.12112254,  1.03292999, -0.18593891,  2.0635165 ,  1.89886391, 2.33205167,  1.55564644,  2.27810902,  1.04673423,  5.80912461, 5.94802799,  0.47924138,  3.48930951,  3.7245207 , -0.19411499, 3.11162961,  4.81651296,  2.44356696,  0.38019858,  4.96813423, 4.04689544,  1.13601224])
N17=np.array([ 2.00436913,  1.78417907,  1.74210424,  1.84214461,  2.64940842, 2.39418337,  1.07222305,  1.36418668,  1.39925293,  1.63079276, 0.67692515,  0.81502885,  2.58996398, -0.19913251,  0.962786  , 2.30972893,  1.15989487,  3.0469801 ,  1.66353522,  3.97101558, 4.90583638,  4.14352197,  5.72956464,  3.32317703,  2.43627923, 0.82896285, -0.06002705,  0.14188743,  1.64464489,  2.11476192, 0.29058238,  4.47643108,  0.71900364,  5.4660928 ])
N18=np.array([ 0.86269289,  1.26950236,  0.17217909,  0.93010726,  1.48735509, 1.74983815,  1.03531111,  1.90086211,  2.66901886,  2.40280508, 1.55049017,  0.69540441,  2.21142026,  0.96767827,  1.63861232, 2.67613222,  2.12302604,  1.83738397,  3.63601148,  1.59216578, 1.62152976,  0.72921984,  4.30473083,  0.5935366 , -0.27482392, 5.99275683,  2.70259706,  4.9948306 ,  5.16953582,  4.92130181, 3.87079768,  2.50706623,  3.21645238,  0.32420958,  1.44079226, 2.35957509])
N19=np.array([ 1.54703824,  2.56476419,  1.73262548,  2.62992262,  2.5019057 , 1.66877878,  2.44079134,  1.86214887,  1.85404762,  1.04871638, 1.95523554,  1.61722966,  0.83202993,  0.9559964 , -0.09728916, 0.97431528,  0.93313901,  0.77967952,  1.68312666,  1.64385444, 3.10294345,  4.78941878,  4.75869301,  0.22796019,  3.17234359, 1.54933543,  2.38425489,  3.97166245,  5.50395728,  5.58602734, 0.04424954,  0.24617647,  2.47228167,  1.34357411,  4.50417257, 3.52501135,  1.36667005,  0.84885578])
N20=np.array([ 2.38672459,  0.05086151,  2.23536628,  0.83530071,  1.38846575, 0.81872529,  0.83020126,  2.286157  ,  1.63357979,  0.81627073, 1.61104878,  2.23922668,  1.85358842,  3.28422574,  1.63893713, 1.48151575,  0.85546225,  2.38237255,  1.60797519,  1.38536292, 4.34632847,  2.39184837,  1.52292857,  3.7317357 ,  3.03021225, 4.86401419,  7.23748457,  5.48658378,  7.11176242,  6.10083309, 4.66839404,  3.15966221,  2.3418832 ,  5.46720027,  3.85611856, 5.48366488,  2.34465298,  6.625157  ,  6.29952722,  1.65544411])
N40=np.array([ 2.5892402 ,  2.56165643,  1.52725453,  3.74231514,  1.54102387, 1.57563523,  0.48181892,  1.27250612,  2.19910846,  1.92154354, 1.55866864,  1.91730985,  0.99592943,  0.92087295,  1.28749831, 1.03980898,  3.13193516,  1.51226732,  1.35101941,  1.05034757, 0.6436039 ,  1.99068373,  2.13250851,  1.90617854,  2.09378886, 1.619081  , -4.65501779,  0.97943963,  2.00080347,  2.55710143, 0.67914099,  1.04917585,  2.11606477,  0.12238914,  2.18987699, 0.62858955,  0.47038361,  1.5074578 , -3.71166562,  1.05776484, 5.22702818,  1.44980382,  1.11594863,  9.66612719,  3.02785865, 3.58395516,  2.8327473 ,  6.34036013,  4.61908787,  4.04827345, 0.55949288,  5.2164146 ,  3.4469612 ,  2.07801977,  4.08383989, 1.40068409,  1.00402858,  2.35601346,  5.21125285,  2.75023043, 4.07025858,  1.43387437,  3.36565638,  0.09763248,  2.06699268, 5.80093426,  4.62620685,  0.69839256,  2.70368429,  3.90936745, 5.20299509,  5.76408722,  0.77514557,  5.14518499, -0.46531698, 0.0632953 ,  1.33395779,  1.78862823,  2.67250229,  4.65419173])

############################################################################################



N=6 ## !! popravi tudi zgoraj v zanki energija !!!
n=10**(2)
sirina=np.sqrt(8*pi/N/np.sqrt(3))
Sirina=np.array([sirina,sirina/2,sirina/5])

#sirina=0.01
#Sirina=np.array([1,0.1,0.01,0.001,0.0001])

th=pi*np.random.rand(N)
fi=np.random.rand(N)*2*pi
#zacetni=[th,fi]    ## zapakirani argumenti v eno spremenljivko
#zacetni=np.append(th,fi)
zacetni=np.ones(2*N)*N6

###### ___minimizacija

Rl_Pot_En=energija(zacetni,N)
print(Rl_Pot_En)

    
#for zz in range(1,6) :
TT=np.array([10,1,0.1,0.01,0.001])
T=1
temp =10**(0)
I=np.empty(0)
SIRINA=np.empty(0)
RANDOM=np.empty(0)

R=np.empty(0)
K=np.empty(0)
L=np.empty(0)
E=np.empty(0)
MM=np.empty(0)
dE=np.empty(0)

#avg_E=np.empty(0)
#avg_dE=np.empty(0)
#avg_DE=np.empty(0)
avg_xt=np.empty(0)
avg_yt=np.empty(0)
avg_zt=np.empty(0)

SuS=np.empty(0)
Cv=np.empty(0)

TEMP=np.empty(0)

dTh=np.empty(0)
dFi=np.empty(0)

ZacetniAnim=np.ones(len(zacetni))*zacetni

c=0
for s in Sirina:
    
    sirina=s
    avg_E=np.empty(0)
    avg_dE=np.empty(0)
    avg_DE=np.empty(0)
    
    for t in TT:
        temp =t
    
        I=np.empty(0)
        SIRINA=np.empty(0)
        RANDOM=np.empty(0)
        
        R=np.empty(0)
        K=np.empty(0)
        L=np.empty(0)
        E=np.empty(0)
        MM=np.empty(0)
        dE=np.empty(0)
        
    
        dTh=np.empty(0)
        dFi=np.empty(0)
    
        l=0
        k=0
        on=0
        for i in range(0,n):
        	
            Nrand = np.random.randint(0,N) ## en elektron fiksen, da ni preveč vrtenja naokrog ## random integers from low (inclusive) to high (exclusive).
            th_rand = np.random.normal(0, sirina)
            fi_rand = np.random.normal(0, sirina)
        
            zacetni2=np.ones(2*N)*zacetni
            zacetni2[Nrand]=zacetni2[Nrand]+th_rand
            zacetni2[N+Nrand]=zacetni2[N+Nrand]+fi_rand
            
            # računanje dE 
            energ=energija(zacetni,N)
            deltaE = energija(zacetni2,N) - energ
          	
            if deltaE <= 0: 
                
                E=np.append(E, energ)
                dE=np.append(dE,deltaE)
                dTh=np.append(dTh,th_rand)
                dFi=np.append(dFi,fi_rand)
                
                k=k+1
                zacetni =  zacetni2
                ZacetniAnim=np.row_stack((ZacetniAnim,zacetni2))
        
        
        #        if abs(energ-wiki[N-2])<=0.01 and on==0:
        #            sirina=sirina/10
        #            on=1
        #            SIRINA=np.append(SIRINA, i+1)
        #            print('on='+str(on))
        #
        #        elif abs(energ-wiki[N-2])<=0.001 and on==1:
        #            sirina=sirina/10
        #            on=2
        #            SIRINA=np.append(SIRINA, i+1)
        #            print('on='+str(on))
        #
        #        elif abs(energ-wiki[N-2])<=0.0001 and on==2:
        #            sirina=sirina/10
        #            on=3
        #            SIRINA=np.append(SIRINA, i+1)
        #            print('on='+str(on))
        #
        #        elif abs(energ-wiki[N-2])<=0.00001 and on==3:
        #            sirina=sirina/10
        #            on=4
        #            SIRINA=np.append(SIRINA, i+1)
        #            print('on='+str(on))
        #
        #        elif abs(energ-wiki[N-2])<=0.000001 and on==4:
        #            sirina=sirina/10
        #            on=5
        #            SIRINA=np.append(SIRINA, i+1)
        #
        #            print('on='+str(on))
        #        elif abs(energ-wiki[N-2])<=0.0000001 and on==5:
        #            sirina=sirina/10
        #            on=6
        #            SIRINA=np.append(SIRINA, i+1)
        #            print('on='+str(on))
          
                I=np.append(I,i+1) # vse izvedene poteze
                
            elif np.random.random() <= np.exp(-deltaE/temp): 
        
                RANDOM=np.append(RANDOM, i+1)
                I=np.append(I,i+1) # vse izvedene poteze
        
        #        th_rand = np.random.normal(0, 0.1)*pi
        #        fi_rand = np.random.normal(0, 0.1)*pi*2
        #    
        #        zacetni2=np.ones(2*N)*zacetni
        #        zacetni2[Nrand]=zacetni2[Nrand]+th_rand
        #        zacetni2[N+Nrand]=zacetni2[N+Nrand]+fi_rand
             
                # računanje dE 
        #        energ=energija(zacetni,N)
        
            
                E=np.append(E, energ)
                dE=np.append(dE,deltaE)
                dTh=np.append(dTh,th_rand)
                dFi=np.append(dFi,fi_rand)
                
                zacetni =  zacetni2   
                ZacetniAnim=np.row_stack((ZacetniAnim,zacetni2))
        
                l=l+1
        
                on=0
        
    
            R=np.append(R,(k+l)/(i+1)) # razmerje med izvedenimi in sprejetimi potezami
            L=np.append(L,l/(i+1)) # razmerje med izvedenimi in sprejetimi potezami
            K=np.append(K,k/(i+1)) # razmerje med izvedenimi in sprejetimi potezami
        
        I=np.append(I,n) # vse izvedene poteze
    #
    #    SIRINA=np.append(SIRINA, 0)
    #    RANDOM=np.append(RANDOM, 0)
    #    
    #    
    #    print('izračun energije po metropolisu za Ne='+str(N)+' pri Temp='+str(temp)+':')
    #    print(energija(zacetni,N))         
    #    print('wiki='+str(wiki[N-2]))         
    #    print(r'dE='+str( abs( wiki[N-2]-E[k-1]) ) )
    #    print('k='+str(k))         
    #    print('l='+str(l))         
    #    
    #    Hfi, xfi, binEdge_fi, Lfi=Hdata(dFi)
    #    Hth, xth, binEdge_th, Lth=Hdata(dTh)
    #    
    #    
    #    TEZISC_x,TEZISC_y,TEZISC_z=Izrac_tezisca(ZacetniAnim)
    #    Y_0_0=Izrac_momenta(ZacetniAnim,0,0) 
    #    Y_1_0=Izrac_momenta(ZacetniAnim,1,0) 
    #    Y_1_1=Izrac_momenta(ZacetniAnim,1,1) 
    #    Y_2_0=Izrac_momenta(ZacetniAnim,2,0) 
    #    Y_2_1=Izrac_momenta(ZacetniAnim,2,1) 
    #    Y_2_2=Izrac_momenta(ZacetniAnim,2,2) 
    #    
        
        #Izrac_povprečja(vektor)
        avgE=np.mean(E)
        avgdE=np.mean(abs(dE))
        avgDE=np.mean(abs(E-np.ones(k+l)*wiki[N-2]))
        
        avg_E=np.append(avg_E,avgE)
        avg_dE=np.append(avg_dE,avgdE)
        avg_DE=np.append(avg_DE,avgDE)
    
    
    
    
    
    F77=plt.figure(77)
    F77=plt.subplot(3, 1, 1 )
    plt.title(r'Temperaturne spremembe povprečja $\left< V_N^{min} \right>_{reached}$ za '+'{:.{}e}'.format(n, 0 )+r' potez pri $N_e$='+str(N))#+' in n='+str(n))#+'(N='+str(N)+',M='+str(M)+')')
    plt.plot(TT,avg_E,color=crta[c], ls='-',marker='o',alpha = 0.95,label=r'$\sigma$='+'{:.{}f}'.format(sirina, 2))
    #plt.axhline(y=wiki[N-2], color='k', linestyle='--',label=r'$V_N^{min}$='+str(wiki[N-2])+' ; (Vir [2])')
    #plt.axhline(y=np.mean(avg_E), color='r', linestyle=':',label=r'$\left< V_N^{min} \right>_{mean}$='+str(np.mean(avgE)))
    plt.xlabel(r'T' ,fontsize=16)   
    plt.ylabel(r'energija' ,fontsize=16)   
    plt.xscale('log')
    plt.yscale('linear')
#    plt.yscale('log')
    plt.legend(loc='best',fontsize=16)
    #
    F77=plt.subplot(3, 1, 2 )
    plt.title(r'Temperaturne spremembe povprečja $\left< |dV_N^{min}| \right>_{reached}$ za '+'{:.{}e}'.format(n, 0 )+r' potez pri $N_e$='+str(N))#+' in n='+str(n))#+'(N='+str(N)+',M='+str(M)+')')
    plt.plot(TT,avg_dE,color=crta[c],ls='-',marker='o',alpha = 0.95,label=r'$\sigma$='+'{:.{}f}'.format(sirina, 2))
    plt.xlabel(r'T' ,fontsize=16)   
    plt.ylabel(r'energija' ,fontsize=16)     
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best',fontsize=16)
    
    F77=plt.subplot(3, 1, 3 )
    plt.title(r'Temperaturne spremembe povprečja $\left< |V_{N}^k-V_{N}^{[2]}| \right>_{reached}$ za '+'{:.{}e}'.format(n, 0 )+r' potez pri $N_e$='+str(N))#+' in n='+str(n))#+'(N='+str(N)+',M='+str(M)+')')
    plt.plot(TT,avg_DE,color=crta[c],ls='-',marker='o',alpha = 0.95,label=r'$\sigma$='+'{:.{}f}'.format(sirina, 2))
    plt.xlabel(r'T' ,fontsize=16)   
    plt.ylabel(r'energija' ,fontsize=16)     
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best',fontsize=16)
    
    c=c+1


###################################### izris grafov ################################################
##
#F100=plt.figure(100)
#
#plt.suptitle(r'Normirani histogrami porazdelitve sprejetih korakov položaja za '+'{:.{}e}'.format(n, 0 )+' potez',fontsize=16)
#F100=plt.subplot(1, 2, 1 ) 
#plt.step(xfi,Hfi,'y',alpha = 0.5)
#plt.xlabel(r'd$\phi$')
#plt.ylabel('n')
##plt.legend(loc='best')
#plt.title(r'porazdelitev $\phi$ korakov ')#+'(N='+str(N)+',M='+str(M)+')')
#
#F100=plt.subplot(1, 2, 2 ) 
#plt.step(xth,Hth,'r',alpha = 0.5)
#plt.xlabel(r'd$\theta$')
#plt.ylabel('n')
##plt.legend(loc='best')
#plt.title(r'porazdelitev $\theta$ korakov ')#+'(N='+str(N)+',M='+str(M)+')')
##plt.tight_layout()
##
##
#
#F50=plt.figure(50)
#
#F50=plt.subplot(3, 1, 1 )
#plt.title(r'Potek energije za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(E,'r',alpha = 0.95,label=r'izračun')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.axhline(y=wiki[N-2], color='k', linestyle='--',label=r'$V_N^{min}$='+str(wiki[N-2])+' ; (Vir [2])')
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'$V_N^{min}$' ,fontsize=16)   
#plt.legend(loc='best',fontsize=16)
#
#F50=plt.subplot(3, 1, 2 )
#plt.title(r'Spremembe energije za vsako od k+l='+str(k+l)+r' sprejetih potez pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(abs(dE),'b',alpha = 0.95,label=r'sprejete $dV_N^{min}$ ')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(abs(E-np.ones(k+l)*wiki[N-2]),'k',alpha = 0.95,label=r'odstopanje $V_N^{min}$ od Vir [2]')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'|d$V_N^{min}$|' ,fontsize=16)     
##plt.xscale('log')
#plt.yscale('log')
#plt.legend(loc='best',fontsize=16)
#
#F50=plt.subplot(3, 1, 3 )
#plt.title(r'Sprejetost potez tekom izvajanja potez $i$=1,..,n')#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(R,'r',alpha = 0.95,label='R=(k+l)/i')#+'{:.{}f}'.format(SHfi, 3 ))
#if k!=0: plt.plot(K,'c',alpha = 0.95,label='K=k/i')#+'{:.{}f}'.format(SHfi, 3 ))
#if l!=0: plt.plot(L,'m',alpha = 0.95,label='L=l/i')#+'{:.{}f}'.format(SHfi, 3 ))
#preklop=0
#randPreklop=0
##for i in I: 
##    if int(i)==int(SIRINA[preklop]):
##        preklop=preklop+1
##        plt.axvline(x=i,color='b')#+'{:.{}f}'.format(SHfi, 3 ))
##    if int(i)==int(RANDOM[randPreklop]):
##        randPreklop=randPreklop+1
##        plt.axvline(x=i,color='g',alpha = 0.3)#+'{:.{}f}'.format(SHfi, 3 ))
##    else: plt.axvline(x=i,color='k',alpha = 0.05)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.xlabel(r'izvedene poteze' ,fontsize=16)   
#plt.ylabel(r'sprejete/izvedene' ,fontsize=16)     
##plt.xscale('log')
##plt.yscale('log')
#plt.legend(loc='best',fontsize=16)
##plt.tight_layout()
#
#
#
#F1000=plt.figure(1000)
#F1000=plt.subplot(1, 2, 1 ) 
#plt.title(r'Potek težišča za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(TEZISC_x,'r:',alpha = 0.95,label=r'$x_T$='+'{:.{}e}'.format(TEZISC_x[-1], 1 ))
#plt.plot(TEZISC_y,'g:',alpha = 0.95,label=r'$y_T$='+'{:.{}e}'.format(TEZISC_z[-1], 1 ))
#plt.plot(TEZISC_z,'b:',alpha = 0.95,label=r'$z_T$='+'{:.{}e}'.format(TEZISC_y[-1], 1 ))
#plt.axhline(y=0, color='k', linestyle='--')
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'vrednost koordinate' ,fontsize=16)   
#plt.legend(loc='best',fontsize=16)
#
#F1000=plt.subplot(1, 2, 2 )
#plt.title(r'Potek momentov $\left<Y_l^m\right>$ za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
##plt.plot(Y_0_0,'k:',alpha = 0.95,label=r'$\left<Y_0^0\right>$='+'{:.{}e}'.format(Y_0_0[-1], 1 ))
##plt.plot(Y_1_0,'r:',alpha = 0.95,label=r'$\left<Y_1^0\right>$='+'{:.{}e}'.format(Y_1_0[-1], 1 ))
##plt.plot(Y_1_1,'b:',alpha = 0.95,label=r'$\left<Y_1^1\right>$='+'{:.{}e}'.format(Y_1_1[-1], 1 ))
##plt.plot(Y_2_0,'g:',alpha = 0.95,label=r'$\left<Y_2^0\right>$='+'{:.{}e}'.format(Y_2_0[-1], 1 ))
##plt.plot(Y_2_1,'y:',alpha = 0.95,label=r'$\left<Y_2^1\right>$='+'{:.{}e}'.format(Y_2_1[-1], 1 ))
##plt.plot(Y_2_2,'m:',alpha = 0.95,label=r'$\left<Y_2^2\right>$='+'{:.{}e}'.format(Y_2_2[-1], 1 ))
#plt.plot(Y_0_0,'k:',alpha = 0.95,label=r'$\left<Y_0^0\right>$='+'{:.{}e}'.format(Y_0_0[0], 1 ))
#plt.plot(Y_1_0,'r:',alpha = 0.95,label=r'$\left<Y_1^0\right>$='+'{:.{}e}'.format(Y_1_0[0], 1 ))
#plt.plot(Y_1_1,'b:',alpha = 0.95,label=r'$\left<Y_1^1\right>$='+'{:.{}e}'.format(Y_1_1[0], 1 ))
#plt.plot(Y_2_0,'g:',alpha = 0.95,label=r'$\left<Y_2^0\right>$='+'{:.{}e}'.format(Y_2_0[0], 1 ))
#plt.plot(Y_2_1,'y:',alpha = 0.95,label=r'$\left<Y_2^1\right>$='+'{:.{}e}'.format(Y_2_1[0], 1 ))
#plt.plot(Y_2_2,'m:',alpha = 0.95,label=r'$\left<Y_2^2\right>$='+'{:.{}e}'.format(Y_2_2[0], 1 ))
#plt.axhline(y=0, color='k', linestyle='--')
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'vrednost momenta' ,fontsize=16)   
#plt.legend(loc='best',fontsize=16)
##plt.tight_layout()
#
#
#



#F77=plt.figure(77)
#F77=plt.subplot(2, 1, 1 )
#plt.title(r'Potek povprečja količin za '+str(k+l)+' sprejetih potez od '+'{:.{}e}'.format(n, 0 )+r' pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(avgE,'r',alpha = 0.95,label=r'$\left< V_N^{min} \right>_{reached}$='+'{:.{}f}'.format(avgE[-1], 9))
#plt.axhline(y=wiki[N-2], color='k', linestyle='--',label=r'$V_N^{min}$='+str(wiki[N-2])+' ; (Vir [2])')
#plt.axhline(y=np.mean(avgE), color='r', linestyle=':',label=r'$\left< V_N^{min} \right>_{mean}$='+str(np.mean(avgE)))
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'$V_N^{min}$' ,fontsize=16)   
#plt.legend(loc='best',fontsize=16)
##
#F77=plt.subplot(2, 1, 2 )
#plt.title(r'Potek povprečja spremembe energije za vsako od k+l='+str(k+l)+r' sprejetih potez pri $N_e$='+str(N)+' in T='+str(temp))#+'(N='+str(N)+',M='+str(M)+')')
#plt.plot(avgdE,'b',alpha = 0.95,label=r' $\left< |dV_N^{min}| \right>_{reached}$='+'{:.{}e}'.format(avgdE[-1], 1 ))
#plt.plot(avgDE,'k',alpha = 0.95,label=r'$\left< |V_{N}^k-V_{N}^{[2]}| \right>_{reached}$='+'{:.{}e}'.format(avgDE[-1], 1 ))
#plt.xlabel(r'sprejete poteze' ,fontsize=16)   
#plt.ylabel(r'|d$V_N^{min}$|' ,fontsize=16)     
##plt.xscale('log')
#plt.yscale('log')
#plt.legend(loc='best',fontsize=16)







###### ____izris_sfer z naboji_______________________________
#
#xN=np.sin(N6[0:N])*np.cos(N6[N:2*N])
#yN=np.sin(N6[0:N])*np.sin(N6[N:2*N])
#zN=np.cos(N6[0:N])
#
#F20=plt.figure(figsize=(10,12))
##F20=plt.subplot(1, 2, 1 ) 
#Axes3D = plt.axes(projection='3d')
#Axes3D.scatter(xN, yN, zN, zdir='z',marker='o', s=10, c='r', depthshade=True)
#plt.title('Začetna minimalna porazdelitev elektronov po enotski krogli ;   N='+str(N))
#
## draw sphere
#u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
#x = np.cos(u)*np.sin(v)
#y = np.sin(u)*np.sin(v)
#z = np.cos(v)
#Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)
#
#plt.xlim([-1.5,1.5])
#plt.ylim([-1.5,1.5])
##plt.zlim([-1.2,1.2])
#plt.tight_layout()
#
#
#
#Th=zacetni[0:N]
#Fi=zacetni[N:2*N]
#
#XN=np.sin(Th)*np.cos(Fi)
#YN=np.sin(Th)*np.sin(Fi)
#ZN=np.cos(Th)
#
#F10=plt.figure(figsize=(10,12))
##figsize=(10,6)
##F20=plt.subplot(1, 2, 2 ) 
#Axes3D = plt.axes(projection='3d')
#Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True)#,label=r'$V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[k+l-1])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[k+l-1]))+'\n d$V_N^k=$'+str(dE[k+l-1])+'\n izvedene poteze $n$='+str(I[k])+'\n sprejete poteze $k$='+str(k))
#plt.title('Porazdelitev elektronov po enotski krogli po '+str(k+l)+' potezah pri T='+str(temp)+' ;   $N_e$='+str(N))
##plt.title(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N)+'\n $V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[k+l-1])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[k+l-1]))+'\n d$V_N^k=$'+str(dE[k+l-1])+'\n izvedene poteze $n$='+str(int(I[k]))+'\n sprejete poteze $k$='+str(k))
##plt.legend(loc=3,fontsize=16)
#
## draw sphere
#u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
#x = np.cos(u)*np.sin(v)
#y = np.sin(u)*np.sin(v)
#z = np.cos(v)
#Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)
#
#plt.xlim([-1.5,1.5])
#plt.ylim([-1.5,1.5])
##plt.zlim([-1.2,1.2])
#plt.tight_layout()




########################################################################################
######## izdelava animacije
################################################
##
#
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=360)
#
#F110=plt.figure(figsize=(8,8))
#
#Axes3D = plt.axes(projection='3d')
##plt.title(r'Porazdelitev elektronov po enotski krogli za minimizacijo energije ;   $N_e$='+str(N))
#    
## draw sphere
#u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:14j]
#x = np.cos(u)*np.sin(v)
#y = np.sin(u)*np.sin(v)
#z = np.cos(v)
#Axes3D.plot_wireframe(x, y, z, color='k',alpha = 0.2)    
#plt.xlim([-1.5,1.5])
#plt.ylim([-1.5,1.5])
#plt.tight_layout()
#
#
#def animate(i):
#
#    polozaj=ZacetniAnim[i]
#    
#    Th=polozaj[0:N]
#    Fi=polozaj[N:2*N]
#    
#    XN=np.sin(Th)*np.cos(Fi)
#    YN=np.sin(Th)*np.sin(Fi)
#    ZN=np.cos(Th)
#
#    TEZ_X= sum(XN)/N
#    TEZ_Y= sum(YN)/N
#    TEZ_Z= sum(ZN)/N
##    plt.suptitle(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N)+'\n $V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[i])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^k=$'+str(dE[i])+'\n izvedene poteze $n$='+str(int(I[i+1]))+'\n sprejete poteze $k$='+str(i+1))        
#    plt.suptitle(r'Porazdelitev elektronov po enotski krogli za minimalno energijo ;   $N_e$='+str(N)+', T='+'{:.{}e}'.format(temp, 1 )+'\n $V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[i])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^k=$'+str(dE[i])+'\n izvedene poteze $n$='+str(int(I[i+1]))+'\n sprejete poteze $k$='+str(i+1))
#    Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True)
#    Axes3D.scatter(TEZ_X, TEZ_Y, TEZ_Z, zdir='z',marker='x', s=10, c='b', depthshade=True)
#
##    Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True,label=r'$V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N=$'+str(E[i])+'\n $|V_N^i-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^i=$'+str(dE[i])+'sprejeta poteza i='+str(i)+ 'od izvedenih n='+str(I[i]))
##    Axes3D.scatter(XN, YN, ZN, zdir='z',marker='o', s=10, c='r', depthshade=True,label=r'$V_N^{min}=$ '+str(wiki[N-2])+'\n $V_N^k=$'+str(E[i])+'\n $|V_N^k-V_N^{min}|=$ '+str(abs( wiki[N-2]-E[i]))+'\n d$V_N^k=$'+str(dE[i])+'\n izvedene poteze $n$='+str(int(I[i+1]))+'\n sprejete poteze $k$='+str(i+1))
##    plt.legend(loc=3,fontsize=16)
#
#ani = matplotlib.animation.FuncAnimation(F110, animate, frames=k+l, repeat=False)
#ani.save('sprejemanje_potez_N'+str(N)+'_T10'+str(T)+'.mp4', writer=writer)

#
##
###





