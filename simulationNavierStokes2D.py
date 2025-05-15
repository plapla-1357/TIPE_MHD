from os import P_NOWAIT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulationMHD import B

nx = 100
ny = 100
nt = 10
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.01

rho = 1
nu = 0.1
I = 1
S = 1
B = 1

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))


def conditionLimites(u, v, p):
    u[0, :] = 0
    u[-1, :] = 0
    
    
def calcule_P(U, v, p):
    pn = p.copy()
    p[1:-1, 1:-1] = ((dy**2 *(pn[2:, 1:-1] + pn[:-2, 1:-1]) + dx**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]))/(2*(dx**2 + dy**2))
                    + rho*dx**2*dy**2/(2*(dx**2 + dy**2)) * () # todo a finir
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



