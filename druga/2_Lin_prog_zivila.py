# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:03:00 2017

@author: jernej

"""
import numpy as np
import matplotlib.pyplot  as plt
import scipy.fftpack
from numpy import loadtxt
from scipy.optimize import linprog

e=2.718281828459045    
pi=3.141592653589793 

barva=['r','b','g','y','m','c','o']

def stolpec(matrix, i):
    return [row[i] for row in matrix]
    
    
DIR = '/home/jernej/Desktop/ModelskaAn/MOJEDELLO/druga/tabela-zivil.dat'



data=[]
for line in open(DIR):
    lines=line.strip()
    if not lines.startswith("#"):
        data.append(lines.split('\t')[1:10])
        for j in range(0,len(data)):
            for i in range(0,len(data[j])):
                data[j][i]=float(data[j][i])
                
zivilo=[]
for line in open(DIR):
    lines=line.strip()
    if not lines.startswith("#"):
        zivilo.append(lines.split('\t')[0])

#data 	energija[kcal]	mascobe[g]	ogljikovi hidrati[g]	proteini[g]	Ca[mg]	Fe[mg]	Vitamin C[mg]	Kalij[mg]  Natrij[mg]
ener=stolpec(data,0)
masc=stolpec(data,1)
oh=stolpec(data,2)
prot=stolpec(data,3)
Ca=stolpec(data,4)
Fe=stolpec(data,5)
VitC=stolpec(data,6)
Ka=stolpec(data,7)
Na=stolpec(data,8)

x=np.ones(48) # pogoj množi količine živil 
y=np.zeros(48) # pogoj množi alkoholne pijače
y[19]=1
y[46]=1
y[47]=1
#c = ener#Minimiziraj število kalorij, če je priporočen minimalni dnevni	vnos:  
#A = [masc, oh, prot, Ca, Fe, VitC, Ka, Na]#, x]#70 g  maščob,  310 g  ogljikovih  hidratov,  50 g  proteinov,  1000 mg  kalcija  ter  18 mg  železa. 
#b = [ 70, 310, 50, 1000, 18, 60, 3500, 2400]#, 20]
#Upoštevaj tudi, da naj dnevni obroki količinsko ne presežejo dveh kilogramov hrane. 

c = masc#Kako se rezultat razlikuje, če minimiziramo vnos maščob?
A = [ener, oh, prot, Ca, Fe, VitC, Ka, Na , x,y] #2000 kcal g  energija,  310 g  ogljikovih  hidratov,  50 g  proteinov,  1000 mg  kalcija  ter  18 mg  železa. 
b = [ 2000, 310, 50, 1000, 18, 60, 3500, 2400, 20,3]

#res = linprog(c, A_eq=A, b_eq=b, options={"disp": True}) 

#Ker rešujemo poenostavljen problem z malo parametri na živilo, so lahko rezultati nerealistični. 
## Z omejitvijo količine posameznih živil v obroku izboljšaš uravnovešenost prehrane
x0_bounds = (0, 2) #Ovseni_kosmici
x1_bounds = (0, 1) #Jabolko
x2_bounds = (0, 1) #Pomfri
x3_bounds = (0, 4) #Govedina
x4_bounds = (0, 4) #Svinjina
x5_bounds = (0, 4) #Piscanec
x6_bounds = (0, 0.5) #Mleko
x7_bounds = (0, 4) #Sir_edamec
x8_bounds = (0, 2) #Kruh_bel
x9_bounds = (0, 2) #Kruh_polnozrnat
x10_bounds = (0, 0.1) #Maslo
x11_bounds = (0, 4) #Skusa_soljena
x12_bounds = (0, 4) #Losos
x13_bounds = (0, 2) #Riz
x14_bounds = (0, 0.2) #Cokolada
x15_bounds = (0, 4) #Fizol
x16_bounds = (0, 1) #Rdeca_pesa
x17_bounds = (0, 1) #Solata
x18_bounds = (0, 1) #Zelje
x19_bounds = (0, 5) #Pivo
x20_bounds = (0, 1) #Grozdje
x21_bounds = (0, 1) #Jagode
x22_bounds = (0, 4) #Jajce
x23_bounds = (0, 2) #Makaroni
x24_bounds = (0, 2) #Torta
x25_bounds = (0, 0.5) #Nutella
x26_bounds = (0, 2) #Krompir
x27_bounds = (0, 1) #Banana
x28_bounds = (0, 0.5) #Kokice
x29_bounds = (0, 1) #Brokoli
x30_bounds = (0, 4) #Tuna
x31_bounds = (0, 2) #Paradižnik
x32_bounds = (0, 1) #Paprika
x33_bounds = (0, 1) #Korenje
x34_bounds = (0, 2) #Strocji_fizol
x35_bounds = (0, 2) #Kumara
x36_bounds = (0, 1) #Pomaranca
x37_bounds = (0, 4) #Puran
x38_bounds = (0, 4) #Mortadela
x39_bounds = (0, 4) #Postrv
x40_bounds = (0, 0.1) #Olivno_olje
x41_bounds = (0, 0.5) #Marmelada
x42_bounds = (0, 0.5) #Med
x43_bounds = (0, 2) #Kakav
x44_bounds = (0, 0.1) #Sol
x45_bounds = (0, 5) #Radenska
x46_bounds = (0, 2) #Vino_cabernet
x47_bounds = (0, 2) #Vino_traminec

res = linprog(c, A_eq=A, b_eq=b, bounds=(x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds,x5_bounds,x6_bounds,x7_bounds,x8_bounds,x9_bounds,x10_bounds,x11_bounds,x12_bounds,x13_bounds,x14_bounds,x15_bounds,x16_bounds,x17_bounds,x18_bounds,x19_bounds,x20_bounds,x21_bounds,x22_bounds,x23_bounds,x24_bounds,x25_bounds,x26_bounds,x27_bounds,x28_bounds,x29_bounds,x30_bounds,x31_bounds,x32_bounds,x33_bounds,x34_bounds,x35_bounds,x36_bounds,x37_bounds,x38_bounds,x39_bounds,x40_bounds,x41_bounds,x42_bounds,x43_bounds,x44_bounds,x45_bounds,x46_bounds,x47_bounds), options={"disp": True})
for i in range(0,len(zivilo)) :
    if res.x[i]>0:
        print(zivilo[i]+'\t\t = '+str(res.x[i]*100)+' g')
        
print('\nskupna masa živil\t = '+str(sum(res.x*100))+' g')
print('skupna energija živil\t = '+str(sum(res.x*ener))+' kcal')
print('skupne maščobe živil\t = '+str(sum(res.x*masc*100))+' g')


#
## import PuLP
#from pulp import *
#
## Create the 'prob' variable to contain the problem data
#prob = LpProblem("The Miracle Worker", LpMaximize)
#
## Create problem variables
#x=LpVariable("Medicine_1_units",0,None,LpInteger)
#y=LpVariable("Medicine_2_units",0, None, LpInteger)
#
## The objective function is added to 'prob' first
#prob += 25*x + 20*y, "Health restored; to be maximized"
## The two constraints are entered
#prob += 3*x + 4*y <= 25, "Herb A constraint"
#prob += 2*x + y <= 10, "Herb B constraint"
#
## The problem data is written to an .lp file
#prob.writeLP("MiracleWorker.lp")
#
## The problem is solved using PuLP's choice of Solver
#prob.solve()
