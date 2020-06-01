import matplotlib.pyplot as plt
import numpy as np

import scipy.fftpack
from scipy import signal
from scipy.fftpack import fftfreq ,fftshift, fft

e=2.718281828459045    

DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/dvanajsta/'

#vrst=19
#stolpec=4218

val = ["lincoln_L30_N00","lincoln_L30_N10","lincoln_L30_N20","lincoln_L30_N30","lincoln_L30_N40"]

#SIG0 = loadtxt(DIR + val[0]+".txt") # branje
text_file0 = open(DIR + val[0]+".txt", "r")
text_file1 = open(DIR + val[1]+".txt", "r")
text_file3 = open(DIR + val[3]+".txt", "r")
lines0=text_file0.read().split(' ')
lines1=text_file1.read().split(' ')
lines3=text_file3.read().split(' ')



vrst=256
stolpec=313

Lincoln0=np.zeros((vrst,stolpec))
Lincoln1=np.zeros((vrst,stolpec))
Lincoln3=np.zeros((vrst,stolpec))
#OUT =np.zeros((vrst,stolpec))
#vLincDEC=np.zeros((vrst,stolpec))
sLincDEC0=np.zeros((vrst,stolpec))
sLincDEC1=np.zeros((vrst,stolpec))
sLincDEC3=np.zeros((vrst,stolpec))

FsLincDEC1=np.zeros((vrst,stolpec))
FsLincDEC3=np.zeros((vrst,stolpec))
FFsLincDEC1=np.zeros((vrst,stolpec))
FFsLincDEC3=np.zeros((vrst,stolpec))
FFFsLincDEC1=np.zeros((vrst,stolpec))
FFFsLincDEC3=np.zeros((vrst,stolpec))



j=0
k=0
for i in range(0,len(lines0)-1):
    
    b0=int(lines0[i].strip('\n'))
    b1=int(lines1[i].strip('\n'))
    b3=int(lines3[i].strip('\n'))
    Lincoln0[j][i-j*313]=b0
    Lincoln1[j][i-j*313]=b1
    Lincoln3[j][i-j*313]=b3
    
    k=k+1

    if k==313:
        k=0
        j=j+1
       
       
FTsum0=np.average(abs(np.fft.fft(Lincoln0[:,1]))[5:250]) ##šum
FTSUM0=np.ones(len(Lincoln0[:,1]))*FTsum0
FTsum1=np.average(abs(np.fft.fft(Lincoln1[:,1]))[5:250]) ##šum
FTSUM1=np.ones(len(Lincoln1[:,1]))*FTsum1
FTsum3=np.average(abs(np.fft.fft(Lincoln3[:,1]))[5:250]) ##šum
FTSUM3=np.ones(len(Lincoln3[:,1]))*FTsum3


    

for i in range(0,stolpec):  
    Tv=np.linspace(1,vrst+1,vrst) ## vektor časovnih točk za plotanje

    sLincDEC0[:,i]= abs(np.fft.ifft(np.fft.fft(Lincoln0[:,i]) / np.fft.fft(e**(-(Tv)/30)/30) ))
    sLincDEC1[:,i]= abs(np.fft.ifft(np.fft.fft(Lincoln1[:,i]) / np.fft.fft(e**(-(Tv)/30)/30) ))
    sLincDEC3[:,i]= abs(np.fft.ifft(np.fft.fft(Lincoln3[:,i]) / np.fft.fft(e**(-(Tv)/30)/30) ))
    
    
    for j in range(0,vrst): 
        if sLincDEC0[j,i]>255: sLincDEC0[j,i]=255 
        if sLincDEC1[j,i]>255: sLincDEC1[j,i]=255 
        if sLincDEC3[j,i]>255: sLincDEC3[j,i]=255
            

    FTSIG0 = np.fft.fft(Lincoln0[:,i])
    FTSIG1 = np.fft.fft(Lincoln1[:,i])
    FTSIG3 = np.fft.fft(Lincoln3[:,i])
    
    FTSIG0_cut =  np.ones(len(FTSIG0))*FTSIG0 ### spektri s porezanim visokofrekvenčnim šumom
    FTSIG1_cut =  np.ones(len(FTSIG1))*FTSIG1
    FTSIG3_cut =  np.ones(len(FTSIG3))*FTSIG3
    
    for k in range(50,len(Lincoln0[:,1])//2+1):  ### prepišem visokofrekvenčni šum
        FTSIG1_cut[k]=FTSIG0[k]
        FTSIG1_cut[-k]=FTSIG0[-k]
        FTSIG3_cut[k]=FTSIG0[k]
        FTSIG3_cut[-k]=FTSIG0[-k]

    FsLincDEC1[:,i]= abs(np.fft.ifft(FTSIG1_cut / np.fft.fft(e**(-(Tv)/30)/30) ))
    FsLincDEC3[:,i]= abs(np.fft.ifft(FTSIG3_cut / np.fft.fft(e**(-(Tv)/30)/30) ))
#    
    FILT1=((abs(np.fft.fft(Lincoln0[:,100]))**2-abs(FTSUM1)**2)/abs(np.fft.fft(Lincoln0[:,100]))**2)
    FILT3=((abs(np.fft.fft(Lincoln0[:,100]))**2-abs(FTSUM3)**2)/abs(np.fft.fft(Lincoln0[:,100]))**2)
#    FILT1=abs((abs(np.fft.fft(Lincoln1[:,100]))**2-abs(FTSUM1)**2)/abs(np.fft.fft(Lincoln1[:,100]))**2)
#    FILT3=abs((abs(np.fft.fft(Lincoln3[:,100]))**2-abs(FTSUM3)**2)/abs(np.fft.fft(Lincoln3[:,100]))**2)

    FFsLincDEC1[:,i]= abs(np.fft.ifft(FILT1* np.fft.fft(Lincoln1[:,i]) / np.fft.fft(e**(-(Tv)/30)/30) ))
    FFsLincDEC3[:,i]= abs(np.fft.ifft(FILT3* np.fft.fft(Lincoln3[:,i]) / np.fft.fft(e**(-(Tv)/30)/30) ))

    FFFsLincDEC1[:,i]= abs(np.fft.ifft(FILT1* FTSIG1_cut / np.fft.fft(e**(-(Tv)/30)/30) ))
    FFFsLincDEC3[:,i]= abs(np.fft.ifft(FILT3* FTSIG3_cut / np.fft.fft(e**(-(Tv)/30)/30) ))


    for j in range(0,vrst): 
        if FsLincDEC1[j,i]>255: FsLincDEC1[j,i]=255 
        if FsLincDEC3[j,i]>255: FsLincDEC3[j,i]=255
        if FFsLincDEC1[j,i]>255: FFsLincDEC1[j,i]=255 
        if FFsLincDEC3[j,i]>255: FFsLincDEC3[j,i]=255

#np.savetxt(DIR+"prenosna.pgm", OUT, fmt='%f',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"Lincoln.pgm", Lincoln0, fmt='%f',delimiter=" ",header="P2\n313 256\n255",comments='')
#np.savetxt(DIR+"vLincolnDEC.pgm", vLincDEC, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"sLincolnDEC0.pgm", sLincDEC0, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"sLincolnDEC1.pgm", sLincDEC1, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"sLincolnDEC3.pgm", sLincDEC3, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')

np.savetxt(DIR+"FsLincolnDEC1.pgm", FsLincDEC1, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"FsLincolnDEC3.pgm", FsLincDEC3, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')

np.savetxt(DIR+"FFsLincolnDEC1.pgm", FFsLincDEC1, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"FFsLincolnDEC3.pgm", FFsLincDEC3, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')

np.savetxt(DIR+"FFFsLincolnDEC1.pgm", FFsLincDEC1, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')
np.savetxt(DIR+"FFFsLincolnDEC3.pgm", FFsLincDEC3, fmt='%d',delimiter=" ",header="P2\n313 256\n255",comments='')




#T=np.linspace(0,len(SIG0),len(SIG0)) ## vektor časovnih točk za plotanje
#freq=np.linspace(0,len(FTSIG0),len(FTSIG0)) ## vektor frekvenčnih točk za plotanje

barva=['k','r','b','y','m','c']

F50=plt.figure(50)
#F50=plt.subplot(2, 1, 1 ) 
#plt.title(r'Izmerjeni časovni poteki $c(t)$',fontsize=16)
#plt.plot(np.fft.fft(Lincoln[:,1]),color=barva[0],ls='-',alpha = 0.95, label=val[0]+r'.dat = $s(t)$')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(SIG1,color=barva[1],ls='-',alpha = 0.75, label=val[1]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(SIG2,color=barva[2],ls='-',alpha = 0.55, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(SIG3,color=barva[3],ls='-',alpha = 0.75, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T,100*e**(-abs(T)/16)/32,color=barva[4],ls=':',alpha = 0.75, label=r'r(t)')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T[0:LS//2],SIG_256,color=barva[1],ls=':',alpha = 0.5, label='N='+str(LS//2)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T[0:LS//4],SIG_128,color=barva[2],ls=':',alpha = 0.5, label='N='+str(LS//4)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##plt.plot(T[0:LS//6],SIG_64,color=barva[3],ls=':',alpha = 0.5, label='N='+str(LS//6)+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
##    plt.xscale('log')
##plt.yscale('log')
#plt.ylabel('Amplituda' ,fontsize=16)  
#plt.xlabel(' t ' ,fontsize=16)
#plt.xlim([0,len(SIG0)])
##    plt.ylim([0,250])
##    plt.title('Umiranje populacije za različne korake in razlicno velika vzorca')#+'(N='+str(N)+',M='+str(M)+')')
#plt.legend(loc=1)



#F50=plt.subplot(2, 1, 2 ) 
plt.title(r'FFT izmerjenih časovnih potekov $C(\omega)$',fontsize=16)
plt.plot(abs(np.fft.fft(Lincoln0[:,-1])),color=barva[0],ls='-',alpha = 0.95, label=val[0]+r'.dat = $S(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(abs((FTSIG1_cut)),color=barva[1],ls='-',alpha = 0.95, label=val[1]+r'.dat = $S(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(abs((FTSIG3_cut)),color=barva[3],ls='-',alpha = 0.95, label=val[3]+r'.dat = $S(\omega)$')#+'{:.{}f}'.format(SHfi, 3 ))
plt.plot(FTSUM0,color=barva[0],ls='-',alpha = 0.75, label='šum '+val[0]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,abs(np.fft.fft(SIG2)),color=barva[2],ls='-',alpha = 0.55, label=val[2]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,abs(np.fft.fft(SIG3)),color=barva[3],ls='-',alpha = 0.75, label=val[3]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSUM0,color=barva[0],ls=':',alpha = 0.95)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSUM1,color=barva[1],ls=':',alpha = 0.75)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSUM2,color=barva[2],ls=':',alpha = 0.55)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSUM3,color=barva[3],ls=':',alpha = 0.75)#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(T,2*abs(np.fft.fft(e**(-abs(T)/16)/32))**2 ,ls=':',alpha = 0.75, label=r'R(t)')#+'{:.{}f}'.format(SHfi, 3 ))
#plt.plot(freq,FTSIG,color=barva[vzorec],ls='-',alpha = 0.95,label=val[vzorec]+'.dat')#+'{:.{}f}'.format(SHfi, 3 ))
#    plt.xscale('log')
plt.yscale('log')
plt.ylabel('PSD($\omega$)' ,fontsize=16)   
plt.xlabel(r'$\omega$' ,fontsize=16)
#plt.xlim([0,vrst])
#    plt.ylim([0,250])
plt.legend(loc=1)