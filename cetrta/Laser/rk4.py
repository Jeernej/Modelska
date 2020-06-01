from numpy import *
#from matplotlib.pyplot import *
#from scipy.integrate import ode
import math

def RK4_step(x, dt, flow):    # iz x(t) dobis x(t + dt)
    n = len(x)
    k1 = [ dt * k for k in flow(x) ]
    x_temp = [ x[i] + k1[i] / 2 for i in range(n) ]
    k2 = [ dt * k for k in flow(x_temp) ]
    x_temp = [ x[i] + k2[i] / 2 for i in range(n) ]
    k3 = [ dt * k for k in flow(x_temp) ]
    x_temp = [ x[i] + k3[i] for i in range(n) ]
    k4 = [ dt * k for k in flow(x_temp) ]
    for i in range(n):
        x[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6

def RK4_adaptive_step(x, dt, flow, accuracy):  # from Numerical Recipes
    SAFETY = 0.9; PGROW = -0.2; PSHRINK = -0.25;
    ERRCON = 1.89E-4; TINY = 1.0E-30
    n = len(x)
    scale = flow(x)
    scale = [ abs(x[i]) + abs(scale[i] * dt) + TINY for i in range(n) ]
    while True:
        
        dt /= 2
        x_half = [ x[i] for i in range(n) ]
        RK4_step(x_half, dt, flow)
        RK4_step(x_half, dt, flow)
        
        dt *= 2
        x_full = [ x[i] for i in range(n) ]
        RK4_step(x_full, dt, flow)
        
        Delta = [ x_half[i] - x_full[i] for i in range(n) ]
        error = max( abs(Delta[i] / scale[i]) for i in range(n) ) / accuracy
        
        if error <= 1:
            break;
        dt_temp = SAFETY * dt * error**PSHRINK
        if dt >= 0:
            dt = max(dt_temp, 0.1 * dt)
        else:
            dt = min(dt_temp, 0.1 * dt)
        if abs(dt) == 0.0:
            raise OverflowError("step size underflow")
            
    if error > ERRCON:
        dt *= SAFETY * error**PGROW
    else:
        dt *= 5
    for i in range(n):
        x[i] = x_half[i] + Delta[i] / 15
#    print(dt)
    return dt

#G_m1_plus_m2 = 4 * math.pi**2

#step_using_y = False

def equations(trv):
    pi=3.1415926535897932384626433
    A=1
    xzun=-5. 
    k=0.1
     
    t = trv[0]
    x = trv[1]
    y = trv[2]
    vx = trv[3]
    vy = trv[4]
    
#    r = math.sqrt(x**2 + y**2)
#    ax = - G_m1_plus_m2 * x / r**3
#    ay = - G_m1_plus_m2 * y / r**3
#    
#    flow = [ 1, vx, vy, ax, ay ]
    f=-k*(x-xzun)+A*sin(2.*pi*(t-10.)/24.)   
    flow = [ t, f, 0 ,0 ,vy ]

#    if step_using_y:            # change independent variable from t to y
#        for i in range(5):
#            flow[i] /= vy
    return flow

def print_data(values):
    names = [ "t", "x", "y", "v_x", "v_y" ]
    units = [ "yr", "AU", "AU", "AU/yr", "AU/yr" ]
    for i in range (5):
        print (" " + names[i] + "\t= " + str(values[i]) + " " + units[i])

def write_data(file, values):
    for value in values:
        file.write(str(value) + "\t")
    file.write("\n")

def integrate(trv, dt, t_max, accuracy, adaptive):
    if adaptive:
        file_name = "kepler_adapt.data"
        f_or_a = "adaptive"
    else:
        file_name = "kepler_fixed.data"
        f_or_a = "fixed"
    file = open(file_name, "w")
    trv = [ 0, 21, 0, 0, 0 ]
    print ("\n Initial conditions:")
    print_data(trv)
    print (" Integrating with " + f_or_a + " step size ...")
    step = 0
    dt_min = dt
    dt_max = dt
    while True:
        write_data(file, trv)
#        y_save = trv[2]
#        global step_using_y     # so next line does not create a local variable
#        step_using_y = False
#        if adaptive==True:
        dt = RK4_adaptive_step(trv, dt, equations, accuracy)
        dt_min = min(dt, dt_min)
        dt_max = max(dt, dt_max)
#        else:
        RK4_step(trv, dt, equations)
        
        t, x[i], y, vx, dt = (trv[i] for i in range(5))
#        if x > 0 and y * y_save < 0:
#            step_using_y = True
#            RK4_step(trv, -y, equations)
#            write_data(file, trv)
#            break
        t=t+dt
        if t > t_max:
            print (" t too big, quitting ...")
            break
        step += 1
    file.close()
    print (" Number of " + f_or_a + " steps = ", step)
    if adaptive:
        print (" Minimum dt = ", dt_min)
        print (" Maximum dt = ", dt_max)
    print_data(trv)
    print (" Trajectory data in", file_name)

#print (" Kepler orbit using fixed and then adaptive Runge-Kutta")
#r_aphelion = float(input(" Enter aphelion distance in AU: "))
#eccentricity = float(input(" Enter eccentricity: "))
#a = r_aphelion / (1 + eccentricity)
#T = a**1.5
#vy0 = math.sqrt(G_m1_plus_m2 * (2 / r_aphelion - 1 / a))
#print (" Semimajor axis a = ", a, " AU")
#print (" Period T = ", T, " yr")
#print (" v_y(0) = ", vy0, " AU/yr")
#dt = float(input(" Enter step size dt: "))
#accuracy = float(input(" Enter desired accuracy for adaptive integration: "))


accuracy =10.**(-5) # desired accuracy
dt0=0.01
t0=0
y0=21
trv = [ t0, y0 , 0, 0, dt0]
integrate(trv, dt0, 100, accuracy,True)
#trv = [ 0, r_aphelion, 0, 0, vy0 ]
#integrate(trv, dt, T, accuracy, True)

