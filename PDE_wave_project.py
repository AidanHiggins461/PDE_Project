import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.animation as anim

N     = 200
t     = 40000
L     = 2*np.pi
k     = 1
del_x = (L/(N-1))
del_t = 0.0005
alpha = 4 * del_t**2 * del_x**-2


avg_n = []
n = np.zeros((t, N)) 
for j in range(len(n[0, :])):
    if j * del_x >= np.pi - 2 and j * del_x <= np.pi + 2:
        n[0, j] += 1


for i in range(1, t):
    for j in range(1, N-1):
        if i == 1:  
            n[i, j] += n[i-1, j] * np.sin(0.5 * j * del_x) * del_t
        else:
            n[i, j] += ( alpha * n[i-1, j+1] 
                        - 2 * (alpha - 1) * n[i-1, j] 
                        + alpha * n[i-1, j-1] 
                        - n[i-2, j])
    avg_n.append(float(np.mean(n[i, :])))

plt.scatter(np.array(range(len(avg_n))) * del_t * np.pi ** -1, avg_n, s = 0.1)
plt.show()

# def plot(X, Y, Z):
    
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
#     # Plot the surface.
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#     ax.view_init(elev = 30, azim = 45)
    
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
    
#     plt.show()

# X = [i*del_x - L/2 for i in range(0, N)]
# Y = [i*del_t for i in range(1, t)]
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = n[1:, :]

# plot(X, Y, Z)

fig, ax = plt.subplots()
current_frame = ax.plot(np.array(range(N)) * del_x, n[0, :])[0]
ax.set(ylim = [-1500, 1500])

def update(frame):
    x = np.array(range(N)) * del_x
    y = n[frame, :]
    current_frame.set_xdata(x)
    current_frame.set_ydata(y)
    return current_frame

anim = anim.FuncAnimation(fig, update, frames=1000, interval = 50)
anim.save('Wave_animation.GIF')