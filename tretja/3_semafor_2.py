# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:30:36 2019

@author: jernej
"""

from __future__ import division
import math as m
import pylab as pl
import numpy as np
from matplotlib import rc

from scipy.optimize import minimize


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig=pl.figure(figsize=(18,8))
p1=fig.add_subplot('121')
p2=fig.add_subplot('122')


def plot(f,x0,xk,p1,v0):
    korak= abs((float(xk)-x0))/10000
    print(korak)
    xarray=[]
    farray=[]
    for i in range (0,10000,1):
        x=float(x0)+korak*i
        xarray.append(x)
        farray.append(f(x,v0)[0])

    p1.plot(xarray,farray)
    p1.text(3,60,"$t_1=$"+str(round(f(0,v0)[2],3))+r"$s$"+"\t \t \t \t $l_0=$"+str(f(0,v0)[1])+"m",fontsize=30)



def f(t):
    lam=0.1811
    C=-1.4
    D=14.0
    return lam*((float(t)**2)/2.0)+C*t+D

def v1(t,v0):
    l0=200
    t1=7.0
    lam=6.0*(l0-t1*v0)/t1**3
    C=3*(l0-t1*v0)/t1**2
    D=v0
    return  (-(lam/2.0)*((t**2)/(2.0))+C*t+D,l0,t1,v0)




plot(v1,0,7,p1,30.0/3.6)
#plot(v1,0,7,p1,60.0/3.6)
plot(v1,0,7,p1,90.0/3.6)
#plot(v1,0,7,p1,120.0/3.6)
#plot(v1,0,7,p1,150.0/3.6)
#plot(v1,0,7,p1,180.0/3.6)
plot(v1,0,7,p1,240.0/3.6)

plot(v1,0,7,p2,30.0/3.6)
#plot(v1,0,7,p1,60.0/3.6)
plot(v1,0,7,p2,90.0/3.6)
#plot(v1,0,7,p1,120.0/3.6)
#plot(v1,0,7,p1,150.0/3.6)
#plot(v1,0,7,p1,180.0/3.6)
plot(v1,0,7,p2,240.0/3.6)

#labels=['',-20*3.6,0*3.6,20*3.6,40*3.6,60*3.6]

lines=p1.get_lines()
pl.setp(lines,linewidth=3)
########################ameba
lines=p2.get_lines()
pl.setp(lines,linewidth=3)
########################ameba
#p1.set_yticklabels(labels)
p1.tick_params(labelsize=20)
p1.set_xlabel(r"\textbf{t(s)}",fontsize=35)
p2.set_xlabel(r"\textbf{t(s)}",fontsize=35)
p1.set_ylabel(r"$\textbf{v(km/h)}$",fontsize=35)

xticklabels=[0,10*3.6,20*3.6,30*3.6,40*3.6,50*3.6,60*3.6,70*3.6]
xticks=np.arange(0,7.5,0.5)
p1.set_xticks(xticks)
p1.set_yticklabels(xticklabels)


xticklabels=[0,10*3.6,20*3.6,30*3.6,40*3.6,50*3.6,60*3.6,70*3.6]
xticks=np.arange(0,7.5,0.5)
p2.set_xticks(xticks)
p2.set_yticklabels(xticklabels)
p2.tick_params(labelsize=20)
p2.set_ylim(0,70)

vmax=200/3.6
vmin=60/3.6
lambdav1=1
lambdav2=2.5

#########################ameba240

v0=240/3.6
delilne_tocke=34
delta_t=7.0/(delilne_tocke)#!!!
v=[]
t=[]
def func(v):
    vsum=((v[0]-v0)/delta_t)**2
    expsum=v0/2
    expmax=m.exp(v0-vmax)
    expmin=m.exp(vmin-v0)
    for i in range(0,delilne_tocke-1,1):
        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
    for i in range(0,delilne_tocke,1):
        if i<(delilne_tocke-1):
            expsum=expsum+v[i]
        else:
            expsum=expsum+v[i]/2
    for i in range(0,delilne_tocke,1):
        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
    return (vsum+1*m.exp(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*m.exp(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t))))

for i in range(0,delilne_tocke,1):
    v.append(i*2)

for i in range(0,delilne_tocke+1,1):
    t.append(i*delta_t)


min=minimize(func,v,method='Powell',callback=None,options={'maxiter':5000,'xtol':0.01,})
v=min.get('x')
vkonc=[v0]
for x in v:
    vkonc.append(x)
p1.plot(t,vkonc,'ro')


delilne_tocke=34
delta_t=7.0/(delilne_tocke)#!!!
v=[]
t=[]
def func(v):
    vsum=((v[0]-v0)/delta_t)**2
    expsum=v0/2
    expmax=m.exp(v0-vmax)
    expmin=m.exp(vmin-v0)
    for i in range(0,delilne_tocke-1,1):
        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
    for i in range(0,delilne_tocke,1):
        if i<(delilne_tocke-1):
            expsum=expsum+v[i]
        else:
            expsum=expsum+v[i]/2
    for i in range(0,delilne_tocke,1):
        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
    return (vsum+1*m.exp(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*m.exp(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+expmin)

for i in range(0,delilne_tocke,1):
    v.append(i*2)

for i in range(0,delilne_tocke+1,1):
    t.append(i*delta_t)


min=minimize(func,v,method='Powell',callback=None,options={'maxiter':5000,'xtol':0.01,})
v=min.get('x')
vkonc=[v0]
for x in v:
    vkonc.append(x)


#########################ameba90

v0=90/3.6
delilne_tocke=34
delta_t=7.0/(delilne_tocke)#!!!
v=[]
t=[]
def func(v):
    vsum=((v[0]-v0)/delta_t)**2
    expsum=v0/2
    expmax=m.exp(v0-vmax)
    expmin=m.exp(vmin-v0)
    for i in range(0,delilne_tocke-1,1):
        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
    for i in range(0,delilne_tocke,1):
        if i<(delilne_tocke-1):
            expsum=expsum+v[i]
        else:
            expsum=expsum+v[i]/2
    for i in range(0,delilne_tocke,1):
        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
    return (vsum+1*m.exp(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*m.exp(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t))))

for i in range(0,delilne_tocke,1):
    v.append(i*2)

for i in range(0,delilne_tocke+1,1):
    t.append(i*delta_t)


min=minimize(func,v,method='Powell',callback=None,options={'maxiter':5000,'xtol':0.01,})
v=min.get('x')
vkonc=[v0]
for x in v:
    vkonc.append(x)
p1.plot(t,vkonc,'go')


delilne_tocke=34
delta_t=7.0/(delilne_tocke)#!!!
v=[]
t=[]
def func(v):
    vsum=((v[0]-v0)/delta_t)**2
    expsum=v0/2
    expmax=m.exp(v0-vmax)
    expmin=m.exp(vmin-v0)
    for i in range(0,delilne_tocke-1,1):
        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
    for i in range(0,delilne_tocke,1):
        if i<(delilne_tocke-1):
            expsum=expsum+v[i]
        else:
            expsum=expsum+v[i]/2
    for i in range(0,delilne_tocke,1):
        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
    return (vsum+1*m.exp(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*m.exp(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+expmin)

for i in range(0,delilne_tocke,1):
    v.append(i*2)

for i in range(0,delilne_tocke+1,1):
    t.append(i*delta_t)


min=minimize(func,v,method='Powell',callback=None,options={'maxiter':5000,'xtol':0.01,})
v=min.get('x')
vkonc=[v0]
for x in v:
    vkonc.append(x)
p2.plot(t,vkonc,'go')


#########################ameba30

v0=30/3.6
delilne_tocke=34
delta_t=7.0/(delilne_tocke)#!!!
v=[]
t=[]
def func(v):
    vsum=((v[0]-v0)/delta_t)**2
    expsum=v0/2
    expmax=m.exp(v0-vmax)
    expmin=m.exp(vmin-v0)
    for i in range(0,delilne_tocke-1,1):
        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
    for i in range(0,delilne_tocke,1):
        if i<(delilne_tocke-1):
            expsum=expsum+v[i]
        else:
            expsum=expsum+v[i]/2
    for i in range(0,delilne_tocke,1):
        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
    return (vsum+1*m.exp(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*m.exp(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t))))

for i in range(0,delilne_tocke,1):
    v.append(i*2)

for i in range(0,delilne_tocke+1,1):
    t.append(i*delta_t)


min=minimize(func,v,method='Powell',callback=None,options={'maxiter':5000,'xtol':0.01,})
v=min.get('x')
vkonc=[v0]
for x in v:
    vkonc.append(x)
p1.plot(t,vkonc,'bo')


delilne_tocke=34
delta_t=7.0/(delilne_tocke)#!!!
v=[]
t=[]
def func(v):
    vsum=((v[0]-v0)/delta_t)**2
    expsum=v0/2
    expmax=m.exp(v0-vmax)
    expmin=m.exp(vmin-v0)
    for i in range(0,delilne_tocke-1,1):
        vsum=vsum+((v[i+1]-v[i])/delta_t)**2
    for i in range(0,delilne_tocke,1):
        if i<(delilne_tocke-1):
            expsum=expsum+v[i]
        else:
            expsum=expsum+v[i]/2
    for i in range(0,delilne_tocke,1):
        expmax=expmax+m.exp(lambdav2*(v[i]-vmax))
        expmin=expmin+m.exp(lambdav2*(vmin-v[i]))
    return (vsum+1*m.exp(lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+1*m.exp(-lambdav1*(expsum-(200.0/delta_t)-(0.5/delta_t)))+expmin)

for i in range(0,delilne_tocke,1):
    v.append(i*2)

for i in range(0,delilne_tocke+1,1):
    t.append(i*delta_t)


min=minimize(func,v,method='Powell',callback=None,options={'maxiter':5000,'xtol':0.01,})
v=min.get('x')
vkonc=[v0]
for x in v:
    vkonc.append(x)
p2.plot(t,vkonc,'bo')




pl.show()