from os import P_NOWAIT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

nx = 41
ny = 41
nt = 10
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.1

rho = 1
nu = 0.1
I = 1
S = 1
B = np.ones((ny, nx)) * 0.1

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.ones((ny, nx))


x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# def conditionLimites(u, v, p):
#     u[0, :] = 0
#     u[-1, :] = 0
#     u[:, 0] = u[:, -2]
#     u[:, -1] = u[:, 1]
    
#     v[:, 0] = v[:, -2]
#     v[:, -1] = v[:, 1]
    
    

    
    
def calcule_P(u, v, p):
    pn = p.copy()
    # Utilisation de dérivées centrées pour toutes les dérivées
    p[1:-1, 1:-1] = (
        (dy**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1]) + dx**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]))
        / (2 * (dx**2 + dy**2))
        + rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * (
            - (1/dt)*((u[:-2, 1:-1] - u[2:, 1:-1]) / (2 * dx) + (v[1:-1, :-2] - v[1:-1, 2:]) / (2 * dy))
            +((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx))**2
            + 2 * (
                (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
                * (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)
            )
            + ((v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy))**2
        )
    )

    # périodicité en y = 0
    p[1:-1, 0] = (
        (dy**2 * (pn[2:, 0] + pn[:-2, 0]) + dx**2 * (pn[1:-1, 1] + pn[1:-1, -1]))
        / (2 * (dx**2 + dy**2))
        + rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * (
            - (1/dt)*((u[:-2, 0] - u[2:, 0]) / (2 * dx) + (v[1:-1, -1] - v[1:-1, 1]) / (2 * dy))
            +((u[2:, 0] - u[:-2, 0]) / (2 * dx))**2
            + 2 * (
                (u[2:, 0] - u[:-2, 0]) / (2 * dx)
                * (v[1:-1, 1] - v[1:-1, -1]) / (2 * dy)
            )
            + ((v[1:-1, 1] - v[1:-1, -1]) / (2 * dy))**2
        )
    )

    # périodicité en y = 2
    p[1:-1, -1] = (
        (dy**2 * (pn[2:, -1] + pn[:-2, -1]) + dx**2 * (pn[1:-1, 0] + pn[1:-1, -2]))
        / (2 * (dx**2 + dy**2))
        + rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * (
            - (1/dt)*((u[:-2, 0] - u[2:, 0]) / (2 * dx) + (v[1:-1, -1] - v[1:-1, 1]) / (2 * dy))
            +((u[2:, -1] - u[:-2, -1]) / (2 * dx))**2
            + 2 * (
                (u[2:, -1] - u[:-2, -1]) / (2 * dx)
                * (v[1:-1, 0] - v[1:-1, -2]) / (2 * dy)
            )
            + ((v[1:-1, 0] - v[1:-1, -2]) / (2 * dy))**2
        )
    )
    
    
    

def calcule_Vitesses(u, v, p):
    un = u.copy()
    u[1:-1, 1: -1] = un[1: -1, 1: -1] + dt * ((p[1: -1, 1:-1] - p[:-2, 1: -1]) / (rho * dx)
                                              -nu*((un[2:, 1: -1] - 2*un[1: -1, 1: -1] + un[:-2, 1: -1])/dx**2 
                                                   + (un[1: -1, 2:] -2*un[1: -1, 1: -1] + un[1: -1, :-2])/dy**2)
                                              - un[1:-1, 1:-1]*(un[1: -1, 1: -1] - un[:-2, 1: -1])/dx
                                              )
    
    # condition de periodicité en y = 0 et y = 2
    u[1: -1, 0] = un[1: -1, 0] + dt * ((p[1: -1, 0] - p[:-2, 0]) / (rho * dx)
                                              -nu*((un[2:, 0] - 2*un[1: -1, 0] + un[:-2, 0])/dx**2 
                                                   + (un[1: -1, 1] -2*un[1: -1, 0] + un[1: -1, -1])/dy**2)
                                              - un[1:-1, 0]*(un[1: -1, 0] - un[:-2, 0])/dx
                                              )
    
    u[1: -1, -1] = un[1: -1, -1] + dt * ((p[1: -1, -1] - p[:-2, -1]) / (rho * dx)
                                              -nu*((un[2:, -1] - 2*un[1: -1, -1] + un[:-2, -1])/dx**2 
                                                   + (un[1: -1, 0] -2*un[1: -1, -1] + un[1: -1, -2])/dy**2)
                                              - un[1:-1, -1]*(un[1: -1, -1] - un[:-2, -1])/dx
                                              )
    
    vn = v.copy()
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * ((p[1:-1, 1:-1] - p[1:-1, :-2])/(rho*dy) 
                                           - nu * ((vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2])/dy**2 
                                                   + (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1])/dx**2)
                                           - vn[1:-1, 1:-1]*(vn[1:-1, 1:-1] - vn[1:-1, :-2])/dy 
                                           + 2*I*B[1: -1, 1:-1]/S
                                           )
    
    # condition de periodicité en y = 0 et y = 2
    v[1:-1, 0] = vn[1:-1, 0] + dt * ((p[1:-1, 0] - p[1:-1, -2])/(rho*dy) 
                                           - nu * ((vn[1:-1, 1] - 2*vn[1:-1, 0] + vn[1:-1, -1])/dy**2 
                                                   + (vn[2:, 0] - 2*vn[1:-1, 0] + vn[:-2, 0])/dx**2)
                                           - vn[1:-1, 0]*(vn[1:-1, 0] - vn[1:-1, -2])/dy 
                                           + 2*I*B[1: -1, 0]/S
                                           )
    
    v[1:-1, -1] = vn[1:-1, -1] + dt * ((p[1:-1, -1] - p[1:-1, -2])/(rho*dy)
                                           - nu * ((vn[1:-1, 0] - 2*vn[1:-1, -1] + vn[1:-1, -2])/dy**2 
                                                   + (vn[2:, -1] - 2*vn[1:-1, -1] + vn[:-2, -1])/dx**2)
                                           - vn[1:-1, -1]*(vn[1:-1, -1] - vn[1:-1, -2])/dy 
                                           + 2*I*B[1: -1, -1]/S
                                           )
    
    return u, v


def simulationNavierStokes():
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx))
    for n in range(nt):
        calcule_P(u, v, p)
        calcule_Vitesses(u, v, p)
        # conditionLimites(u, v, p)
    
    fig = plt.figure(figsize=(11,7), dpi=100)
    # plotting the pressure field as a contour
    plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    plt.colorbar()
    # plotting the pressure field outlines
    plt.contour(X, Y, p, cmap=cm.viridis)  
    # plotting velocity field
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    # plt.streamplot(X, Y, u, v)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
        
    
simulationNavierStokes()



