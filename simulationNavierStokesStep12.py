from re import I
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import simulation



##### champs mag #####
ChampMag = {
    "X": [1.5]*19 + [1]*19 + [2]*19 + [0.5]*19,
    "Y": [-1.0, -0.5, 0, 0.5, 1., 1.5, 2., 2.5, 3, 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.] * 4,
    "B": [
        0.6, 3.5, 7.7, 13.8, 17.1, 16.9, 15.2, 15.5, 17.1, 18.3, 17, 15.7, 17, 19, 18.9, 13.9, 8.2, 3.9, 1.8,
        0.6, 3.6, 8.5, 14.2, 17.7, 17.6, 16, 17, 18.8, 19.6, 18.1, 16.5, 17.3, 19, 18.1, 12.4, 7.4, 3.7, 1,
        0.2, 1.8, 4.8, 8.8, 11.1, 10.8, 9.6, 9.5, 10.3, 10.6, 10, 9.4, 10, 11.2, 11, 8.5, 4.5, 2.1, 0.6,
        0.8, 3.4, 8.1, 13.3, 15.8, 15.5, 14.1, 15, 17, 17.5, 15.9, 14.5, 15.6, 16.8, 16, 11.2, 5.9, 2.6, 0.7
    ]
}

def get_real_champs_mag(x, y):
    return ChampMag["B"][ChampMag["X"].index(x) + ChampMag["Y"].index(y)]


def get_champs_mag6(x, y):
    # Vérifie si le point exact existe dans les données
    if (x, y) in zip(ChampMag["X"], ChampMag["Y"]):
        index_x = ChampMag["X"].index(x)
        index_y = ChampMag["Y"].index(y)
        return ChampMag["B"][index_x + index_y]  

    # Interpolation sur X uniquement
    if x in ChampMag["X"]:
        y1, y2 = np.floor(y * 2) / 2, np.ceil(y * 2) / 2
        if y1 == y2:  # Évite une division par zéro
            return get_real_champs_mag(x, y1)
        B1 = get_real_champs_mag(x, y1)
        B2 = get_real_champs_mag(x, y2)
        t = (y - y1) / (y2 - y1)
        return B1 * (1 - t) + B2 * t

    # Interpolation sur Y uniquement
    if y in ChampMag["Y"]:
        x1, x2 = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
        if x1 == x2:  # Évite une division par zéro
            return get_real_champs_mag(x1, y)
        B1 = get_real_champs_mag(x1, y)
        B2 = get_real_champs_mag(x2, y)
        t = (x - x1) / (x2 - x1)
        return B1 * (1 - t) + B2 * t

    # Interpolation triangulaire
    if (2 * x) % 1 + (2 * y) % 1 < 1:  # Triangle inférieur
        x1, x2, x3 = np.floor(x * 2) / 2, np.ceil(x * 2) / 2, np.floor(x * 2) / 2
        y1, y2, y3 = np.floor(y * 2) / 2, np.floor(y * 2) / 2, np.ceil(y * 2) / 2
    else:  # Triangle supérieur
        x1, x2, x3 = np.ceil(x * 2) / 2, np.ceil(x * 2) / 2, np.floor(x * 2) / 2
        y1, y2, y3 = np.floor(y * 2) / 2, np.ceil(y * 2) / 2, np.ceil(y * 2) / 2

    # Champs magnétiques aux sommets du triangle
    B1 = get_real_champs_mag(x1, y1)
    B2 = get_real_champs_mag(x2, y2)
    B3 = get_real_champs_mag(x3, y3)

    # Interpolation barycentrique
    denom = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if denom == 0:  # Évite une division par zéro
        return (B1 + B2 + B3) / 3  

    lambda1 = ((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)) / denom
    lambda2 = ((x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)) / denom
    lambda3 = 1 - lambda1 - lambda2

    # Calcul du champ magnétique interpolé
    B = lambda1 * B1 + lambda2 * B2 + lambda3 * B3
    return B






def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    
    return b


def pressure_poisson_periodic(p, dx, dy):
    pn = np.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p


nx = 41
ny = 41
nt = 20
nit = 2000
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
y = np.linspace(-1, 8, nx)
x = np.linspace(0.5, 2, ny)
X, Y = np.meshgrid(x, y)


##physical variables
rho = 1
nu = .1
I = 3
S = 1

B = np.zeros_like(X)
for i in range(len(y)):
    for j in range(len(x)):
        B[i, j] = get_champs_mag6(x[j], y[i])


Fy = I/S*B
# Fy = 1
Fx = 0
dt = .01

#initial conditions
u = np.zeros((ny, nx))
un = np.zeros((ny, nx))

v = np.zeros((ny, nx))
vn = np.zeros((ny, nx))

p = np.ones((ny, nx))
pn = np.ones((ny, nx))

b = np.zeros((ny, nx))
udiff = 1
stepcount = 0

while udiff > .001:
    un = u.copy()
    vn = v.copy()

    b = build_up_b(rho, dt, dx, dy, u, v)
    p = pressure_poisson_periodic(p, dx, dy)

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx * 
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy * 
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                    dt / (2 * rho * dx) * 
                    (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                    nu * (dt / dx**2 * 
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 * 
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                    Fy[1: -1, 1: -1] * dt)

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx * 
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 * 
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])) + Fx * dt)

    # Periodic BC u @ x = 2     
    u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                (un[1:-1, -1] - un[1:-1, -2]) -
                vn[1:-1, -1] * dt / dy * 
                (un[1:-1, -1] - un[0:-2, -1]) -
                dt / (2 * rho * dx) *
                (p[1:-1, 0] - p[1:-1, -2]) + 
                nu * (dt / dx**2 * 
                (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                dt / dy**2 * 
                (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + Fy[1: -1, -1] * dt)

    # Periodic BC u @ x = 0
    u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (un[1:-1, 0] - un[1:-1, -1]) -
                vn[1:-1, 0] * dt / dy * 
                (un[1:-1, 0] - un[0:-2, 0]) - 
                dt / (2 * rho * dx) * 
                (p[1:-1, 1] - p[1:-1, -1]) + 
                nu * (dt / dx**2 * 
                (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                dt / dy**2 *
                (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + Fy[1: -1, 0] * dt)

    # Periodic BC v @ x = 2
    v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                (vn[1:-1, -1] - vn[1:-1, -2]) - 
                vn[1:-1, -1] * dt / dy *
                (vn[1:-1, -1] - vn[0:-2, -1]) -
                dt / (2 * rho * dy) * 
                (p[2:, -1] - p[0:-2, -1]) +
                nu * (dt / dx**2 *
                (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                dt / dy**2 *
                (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])) + Fx * dt)

    # Periodic BC v @ x = 0
    v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (vn[1:-1, 0] - vn[1:-1, -1]) -
                vn[1:-1, 0] * dt / dy *
                (vn[1:-1, 0] - vn[0:-2, 0]) -
                dt / (2 * rho * dy) * 
                (p[2:, 0] - p[0:-2, 0]) +
                nu * (dt / dx**2 * 
                (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                dt / dy**2 * 
                (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])) + Fx * dt)


    # Wall BC: u,v = 0 @ y = 0,2
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :]=0
    
    udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
    stepcount += 1
    
fig = plt.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
plt.colorbar()
# plotting the pressure field outlines
plt.contour(X, Y, p, cmap=cm.viridis)  
# plotting velocity field
# plt.quiver(X[::2, ::2], Y[::2, ::2], v[::2, ::2], u[::2, ::2]) 
plt.streamplot(X, Y, v, u)
plt.xlabel('X')
plt.ylabel('Y')
simulation.show3DMap()



plt.show()
        
