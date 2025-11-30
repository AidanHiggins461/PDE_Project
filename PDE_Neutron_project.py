import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

Diffusion_a  = False
Analytical_a = False
Diffusion_b  = False
Diffusion_c  = False
Diffusion_d  = False


N     = 100
T     = 30000
D     = 1*10**0
C     = 1*10**0
alph  = 1
L = np.pi
del_x = ((alph)*L/(N-1))
del_t = 0.001
alpha = (D*del_t)/((del_x)**2)
n_0   = (del_x)**(-1)
strt = [0 , np.pi+0.1]
err = 1e-10
avg_n = []
n = np.zeros((T, N))


def bisection(func, N, T, strt, err, C, D, alph):
    a  = strt[0]
    b =strt[1]
    m = [1e3, 1]
    while abs(m[0]) > err:
        length = 0.5 * (a+b)
        L  = length*np.sqrt(D)/(np.sqrt(C))
        del_x = ((alph)*L/(N-1))
        del_t = del_x**2 * 0.49 * D
        alpha = (D*del_t)/((del_x)**2)
        n_0   = 10
        avg_n = []
        n = np.zeros((T, N)) 
        n[0, int(N/2)] += n_0
        n, m = func(n, N, T, del_t, strt, alph, alpha)
        if m[0] > 0:
            b = L
        else:
            a = L
        print('length:', L, 'm:', m)
    return n, del_x, del_t, L

def diffusion_00(n, N, T, del_t, strt, alph, alpha):
    avg_n = []
    for i in range(1, T):
        for j in range(1, N-1):
                    n[i, j] += (del_t*C+1)*n[i-1, j] + alpha*(n[i-1, j+1] - (2)*n[i-1, j] + n[i-1, j-1])
        avg_n.append(float(np.mean(n[i, :])))
    m = np.polyfit(range(len(avg_n))[int(T*0.9):], avg_n[int(T*0.9):], deg = 1)
    return n, m

def diffusion_0d(n, N, T, del_t, strt, alph, alpha):
    avg_n = []
    for i in range(1, T):
        for j in range(1, N-1):
                n[i, j] += (del_t+C)*n[i-1, j] + alpha*(n[i-1, j+1] - (2)*n[i-1, j] + n[i-1, j-1])
        n[i, 0] += n[i, 1]
        avg_n.append(float(np.mean(n[i, :])))
    m = np.polyfit(range(len(avg_n))[int(T*0.9):], avg_n[int(T*0.9):], deg = 1)
    return n, m

def diffusion_dd(n, N, T, del_t, strt, alph, alpha):
    avg_n = []
    for i in range(1, T):
        for j in range(1, N-1):
                    n[i, j] += (del_t+C)*n[i-1, j] + alpha*(n[i-1, j+1] - (2)*n[i-1, j] + n[i-1, j-1])
        n[i, 0] += n[i, 1]
        n[i, N-1] += n[i, N-2]
        avg_n.append(float(np.mean(n[i, :])))
    m = np.polyfit(range(len(avg_n))[int(T*0.9):], avg_n[int(T*0.9):], deg = 1)
    return n, m

def diffusion_alpha(n, N, T, del_t, strt, alph, alpha):
    avg_n = []
    for i in range(1, T):
        for j in range(1, N-1):
            if abs(j*del_x - alph*L/2) < L/2:
                n[i, j] += (del_t*C+1)*n[i-1, j] + alpha*(n[i-1, j+1] - (2)*n[i-1, j] + n[i-1, j-1])
            else:
                n[i, j] += n[i-1, j] + alpha*(n[i-1, j+1] - (2)*n[i-1, j] + n[i-1, j-1])
        avg_n.append(float(np.mean(n[i, :])))
    m = np.polyfit(range(len(avg_n))[int(T*0.9):], avg_n[int(T*0.9):], deg = 1)
    return n, m


def analytical_00(L, N, T, C, D):
    xs = np.linspace(-L/2, L/2, N) 
    ts = np.linspace(0, del_t*T, T)
    xs,ts = np.meshgrid(xs, ts)  
    n_an = np.zeros((T,N))
    Nsum = 100
    
    for m in range(1, Nsum + 1):
                n_an +=   2 * L**-1 * (np.cos((2*m-1) * np.pi * xs * L**-1) 
                                        * np.exp(ts * (C - (np.pi * D *  (2*m- 1) * L**-1)**2)))
    return xs, ts, n_an
            
    

def plot(X, Y, Z, L, alph):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.xlabel('x')
    plt.ylabel('t')
    ax.view_init(elev = 30, azim = 45)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f'alpha={alph} L={L}')
    
    plt.show()


n, del_x, del_t, L = bisection(diffusion_0d, N, T, strt, err, C, D, alph)
xs, ts, n_an = analytical_00(L, N, T, C, D)


X = [i*del_x - alph*L/2 for i in range(0, N)]
Y = [i*del_t for i in list(range(0, T))]
X, Y = np.meshgrid(X, Y)
Z = n[:, :]

plot(X, Y, Z, L, alph)



