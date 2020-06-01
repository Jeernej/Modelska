import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(0)

hi = [0]*19

temp = 0.01

#print hi

for i in range(0,100000):
	p = np.random.randint(1,18)
	hinew = -100.0*np.random.random()
	
	deltaE = hinew - hi[p] + 0.5*( (hinew - hi[p-1])**2 - (hi[p] - hi[p-1])**2 + (hinew - hi[p+1])**2 - (hi[p] - hi[p+1])**2 )
	
	if deltaE <= 0: hi[p] = hinew
	elif np.random.random() <= np.exp(-deltaE/temp): hi[p] = hinew
		
#print hi

fig = plt.figure(figsize=(12,10)) 
ax =fig.add_subplot(111)

x=np.arange(19)
ax.plot(x,hi)

plt.show()

