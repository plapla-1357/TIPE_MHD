from os import P_NOWAIT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nx = 100
ny = 100
nt = 200
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.01

rho = 1
nu = 0.1
I = 1
S = 1
B = np.ones((ny, nx)) * 0.1

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))


x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

def conditionLimites(u, v, p):
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    
    v[:, 0] = v[:, -2]
    v[:, -1] = v[:, 1]
    
    
    p[:, -1] = p[:, -2] # periodicité
    p[:, 0] = p[:, 1]
    

    
    
def calcule_P(U, v, p):
    pn = p.copy()
    p[1:-1, 1:-1] = ((dy**2 *(pn[2:, 1:-1] + pn[:-2, 1:-1]) + dx**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]))/(2*(dx**2 + dy**2))
                    + rho*dx**2*dy**2/(2*(dx**2 + dy**2)) * (((u[1:-1, 1:-1] - u[:-2, 1:-1])/dx)**2 
                                                             + 2*((u[1:-1, 1:-1] - u[:-2, 1:-1])*(v[1:-1, 1:-1] - v[1:-1, :-2])/(dx*dy)) 
                                                             + ((v[1:-1, 1:-1] - v[1:-1, :-2])/dy)**2 ) # todo a finir
                    )
    

def calcule_V(u, v, p):
    un = u.copy()
    u[1:-1, 1: -1] = un[1: -1, 1: -1] + dt * ((p[1: -1, 1:-1] - p[:-2, 1: -1]) / (rho * dx)
                                              -nu*((un[2:, 1: -1] - 2*un[1: -1, 1: -1] + un[:-2, 1: -1])/dx**2 
                                                   + (un[1: -1, 2:] -2*un[1: -1, 1: -1] + un[1: -1, :-2])/dy**2)
                                              - un[1:-1, 1:-1]*(un[1: -1, 1: -1] - un[:-2, 1: -1])/dx
                                              )
    
    vn = v.copy()
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * ((p[1:-1, 1:-1] - p[1:-1, :-2])/(rho*dy) 
                                           - nu * ((vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])/dy**2 
                                                   + (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])/dx**2)
                                           - vn[1:-1, 1:-1]*(vn[1:-1, 1:-1] - vn[1:-1, :-2])/dy 
                                           + 2*I*B[1: -1, 1:-1]/S
                                           )
    
    return u, v


def simulationNavierStokes():
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    for n in range(nt):
        calcule_P(u, v, p)
        calcule_V(u, v, p)
        conditionLimites(u, v, p)
    
    fig = plt.figure(figsize = (11,7), dpi=100)
    plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
    plt.show()
        
    
simulationNavierStokes()



