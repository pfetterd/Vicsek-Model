#====================================================#
#	Modelo de Viczek
#====================================================#

import numpy as np
from numpy.linalg import *
import matplotlib
matplotlib.use('TkAgg') 
import math as m
import matplotlib.pyplot as plt
from pylab import *

fig = plt.figure() #Usado para animações
ax = fig.add_subplot(111)

def init():
	global R,L,N,dt,tmax,eta,v
	R=1
	N=60
	L=10
	dt=0.05
	tmax=500
	eta=1.
	pi=np.pi
	V=1. #velocidade absoluta


# Posições e velocidades iniciais 

	x = np.random.uniform(0,L,N)
	y = np.random.uniform(0,L,N)
	theta=np.random.normal(-pi,pi,N)#Distribuição aleatoria 0 a 2pi
	#print(theta)		
	vx=V*np.cos(theta)
	vy=V*np.sin(theta) 
		
	itmax=int(tmax/dt)
	return R,eta,V,L,N,dt,tmax,x,y,vx,vy,theta,itmax

def animate():

	class particle():
		def __init__(self, x,y, vx,vy,theta,ident,N):
			self.r = np.array([x,y])
			self.v = np.array([vx,vy])
			self.theta = theta
			self.ident=ident
			self.mid_theta = 0.
			
	
		def evol_r(self,dt): #Evolução temporal
			self.r+=self.v*dt
	
		def evol_theta(self):
			dtheta=np.random.uniform(-eta,eta)
			self.theta = self.mid_theta + dtheta
	
		def velocidade(self,vx,vy):
			vx = V*m.cos(self.theta)
			vy = V*m.sin(self.theta)
			self.v = np.array([vx,vy])
			

		def reflex(self,L):
			if self.r[0] <= 0 : self.v[0]=abs(self.v[0])
			if self.r[0] >= L : self.v[0]=-abs(self.v[0])
			if self.r[1] <= 0 : self.v[1]=abs(self.v[1])
			if self.r[1] >= L : self.v[1]=-abs(self.v[1])

		def reflex1(self,L):
			if self.r[0] <= 0 : self.r[0]=L-.05
			if self.r[0] >= L : self.r[0]=0
			if self.r[1] <= 0 : self.r[1]=L-.05
			if self.r[1] >= L : self.r[1]=0
		

		def mid_angle(self):
			seno,coseno=0., 0.
			cont=0
			for i in range(N):
				if(i!=self.ident):
					dr=np.linalg.norm(self.r - part[i].r)
					if(dr<=R):
						seno+=m.sin(part[i].theta)
						coseno+=m.cos(part[i].theta)
						cont+=1
			if(cont!=0): 
				seno=seno/cont
				coseno=coseno/cont
				self.mid_theta=m.atan(seno/coseno)
			if(cont==0): self.mid_theta=0.
			#print(self.mid_theta)

	R,eta,V,L,N,dt,tmax,x,y,vx,vy,theta,itmax = init()
	part = list(particle(x[i],y[i],vx[i],vy[i],theta[i],i,N) for i in range(N))
	
	ax.set_autoscale_on(False)
	line,= ax.plot(x,y,'^', c='black')
	t=0
	
	for it in range(itmax):
		x=[]
		y=[]
		vx=[]
		vy=[]
		
		#if(it%10==0): eta+=.005
		
		list(map(lambda i:i.mid_angle(), part))
		list(map(lambda i:i.evol_theta(), part))
		list(map(lambda i:i.velocidade(vx,vy), part))
		list(map(lambda i:i.reflex(L), part))
		list(map(lambda i:i.evol_r(dt), part))	
		list(map(lambda i:i.reflex1(L), part))

		
		list(map(lambda i:x.append(i.r[0]), part))
		list(map(lambda i:y.append(i.r[1]), part))
		
		qui=0.
		for i in range(N):
			qui+=np.linalg.norm(part[i].v)*m.cos(part[i].theta)
		qui=qui/(V*N)		


		line.set_data(x,y)
		t+=dt
		ax.set_title('Modelo de Vicsek \n rho = %.2f eta = %.1f  V = %.1f \nt = %.2f X = %.2f'%(N/L/L, eta, V, t,qui))
		fig.canvas.draw()
	return

R,eta,V,L,N,dt,tmax,x,y,vx,vy,theta,itmax = init()
win = fig.canvas.manager.window
fig.canvas.manager.window.after(1, animate)
plt.axis([0,L,0,L])
plt.show()


