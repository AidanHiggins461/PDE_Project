import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as anim

numerical = True
SepofVar  = True
Dalembert = True

N     = 300
T     = 10000
L     = 2*np.pi
k     = 1
del_x = (L/(N-1))
del_t = 0.001
alpha = 4 * del_t**2 * del_x**-2

u = np.zeros((T, N)) 

if numerical:
    for j in range(len(u[0, :])):
        if j * del_x >= np.pi - 2 and j * del_x <= np.pi + 2:
            u[0, j] += 1
    
    for i in range(1, T):
        for j in range(1, N-1):
            if i == 1:  
                u[i, j] += u[i-1, j] + np.sin(0.5 * j * del_x) * del_t
            else:
                u[i, j] += ( alpha * u[i-1, j+1] 
                            - 2 * (alpha - 1) * u[i-1, j] 
                            + alpha * u[i-1, j-1] 
                            - u[i-2, j])

if SepofVar:
    xs = np.linspace(0, 2*np.pi , N) 
    ts = np.linspace(0, del_t*T ,T)
    xs,ts = np.meshgrid(xs, ts)  
    u_s = np.zeros((T,N))
    Nsum = 100
    
    for n in range(1, Nsum + 1):
        if n ==1:
            u_s +=   np.sin(0.5*xs*n) * (4/(n*np.pi) * np.sin(n*np.pi*0.5)*np.sin(n)*np.cos(n*ts) +  (np.sin(ts)))
        else:
            u_s +=   np.sin(0.5*xs*n) * (4/(n*np.pi) * np.sin(n*np.pi*0.5)*np.sin(n)*np.cos(n*ts))
       

if Dalembert:
    L = 6*np.pi
    del_x = L/(N-1)
    del_t = 0.001
    u_D = np.zeros((T,N))
    for i in range(0, T):
        for j in range(1, N-1):     
            
            x = j*del_x - 2*np.pi
            t = i*del_t
            F1 = 0
            F2 = 0
            if np.pi + 2*(t-1) < x and x < np.pi + 2*(t+1):
                F1 += 1
            else:
                F1 += 0
            if np.pi - 2*(t+1) < x and x < np.pi + 2*(-t+1):
                F2 += 1
            else:
                F2 += 0
                
            if 3*np.pi + 2*(t-1) < x and x < 3*np.pi + 2*(t+1):
                F1 += -1
            else:
                F1 += 0
            if 3*np.pi - 2*(t+1)< x and x < 3*np.pi + 2*(-t+1):
                F2 += -1
            else:
                F2 += 0
            if -1*np.pi + 2*(t-1) < x and x < -1*np.pi + 2*(t+1):
                F1 += -1
            else:
                F1 += 0
            if -1*np.pi - 2*(t+1) < x and x < -1*np.pi + 2*(-t+1):
                F2 += -1
            else:
                F2 += 0
            u_D[i, j] +=   0.5*(F1 + F2) + np.sin(0.5*x)* np.sin(t)
        

def plot(N, T, numerical, u, SepofVar, u_s, Dalembert, u_D):
    if numerical:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title('FInite Difference')
        X = [i*del_x for i in range(0, N)]
        Y = [i*del_t for i in range(1, T)]
        X, Y = np.meshgrid(X, Y)
        Z = u[1:, :]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.view_init(elev = 30, azim = 45)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel('x')
        plt.ylabel('t')
        
        plt.show()
    
    if SepofVar:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title('Separation of variables')
        X = [i*del_x for i in range(0, N)]
        Y = [i*del_t for i in range(0, T)]
        X, Y = np.meshgrid(X, Y)
        Z = u_s[:, :]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.view_init(elev = 30, azim = 45)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel('x')
        plt.ylabel('t')
        
        plt.show()
    if Dalembert:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title("D'Alembert's Formula")
        X = [i*del_x - 2*np.pi for i in range(int(N*1/3), int(N*2/3))]
        Y = [i*del_t for i in range(0, T)]
        X, Y = np.meshgrid(X, Y)
        Z = u_D[:, int(N*1/3):int(N*2/3)]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.view_init(elev = 30, azim = 45)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel('x')
        plt.ylabel('t')
        
        plt.show()
    
plot(N, T, numerical, u, SepofVar, u_s, Dalembert, u_D)

if Numerical and SepofVar and Dalembert and True:
    for i in range(100):
        plt.plot([i*del_x for i in range(0, N)], u[i*100, :], label = 'FD')
        plt.plot([i*del_x for i in range(0, N)], u_s[i*100, :], label = 'SoV')
        plt.plot([3*i*del_x - 6*np.pi for i in range(int(N*1/3), int(N*2/3))], u_D[i*100, int(N*1/3):int(N*2/3)], label = "D'A")
        plt.legend()
        plt.grid()
        plt.show()

