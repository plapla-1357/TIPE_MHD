import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
----------------------------------------------------

            ^
            | j          
            |

----------------------------------------------------> x



"""

# constantes physique

rho = 1e3  # kg/m^3
mu = 1e-3  # Pa.s
nu = 1e-6  # m^2/s

l = 10e-2  # m
h = 4e-3 #m
s = l * h
e = 1.5e-2 # m
V = 300e-6 #m3
# L = V/(e*h)
L = 50e-2

B0 = 11e-3  # T
I = 1 # A
j = I/s 



# parametres simulation
nx = 100
ny = 40
nt = 100000
dx = L / nx
dy = e / ny
dt = 0.001

x = np.linspace(0, L, nx)
y = np.linspace(0, e, ny)


# champs magnétique
# 1: uniforme
# 2: circulaire

Btype = 1
if Btype == 1:
    B = np.ones((nx, ny)) * B0
elif Btype == 2:
    B = np.zeros((nx, ny)) 
    B[0:int((nx*l)//L), :] = B0




def updateV(v):
    vn = v.copy()
    v[1: -1, 1: -1] = (vn[1: -1, 1: -1] 
                       + dt* (nu * (vn[2:, 1: -1] - 2*vn[1: -1, 1: -1] + vn[0: -2, 1: -1]) / dx**2
                       + nu * (vn[1: -1, 2:] - 2*vn[1: -1, 1: -1] + vn[1: -1, 0: -2]) / dy**2
                       + j*B[1: -1, 1: -1]/rho
                       )
    ) 
    
    #conditions limites
    v[:, 0] = 0
    v[:, -1] = 0
    
    #condition de periodicité
    v[0, 1: -1] = (vn[0, 1: -1]
                   + dt*(nu*(vn[1, 1: -1] - 2*vn[0, 1: -1] + vn[-1, 1: -1])/dx**2
                         +nu*(vn[0, 2:] -2*vn[0, 1: -1] + vn[0, :-2])/dy**2
                         +j*B[0, 1:-1] / rho
                         )
                   )
    v[-1, 1: -1] = (vn[0, 1: -1]
                   + dt*(nu*(vn[0, 1: -1] - 2*vn[-1, 1: -1] + vn[-2, 1: -1])/dx**2
                         +nu*(vn[-1, 2:] -2*vn[-1, 1: -1] + vn[-1, :-2])/dy**2
                         +j*B[-1, 1:-1]/rho
                         )
                   )
    
    return v

def simulation():
    X, Y = np.meshgrid(x, y)
    v = np.zeros((nx, ny))  # vitesse selon x
    u = np.zeros((nx, ny))  # vitesse selon y (nulle ici)

    for i in range(nt):
        v = updateV(v)
        if i % 100 == 0:
            vmax = np.max(np.abs(v))  # Vitesse maximale (valeur absolue)
            print(f"t = {i*dt:.2f} s — Vitesse max = {vmax:.4f} m/s")

            plt.clf()
            plt.imshow(v.T, cmap='jet', origin='lower', extent=[0, L, 0, e])
            plt.colorbar(label='Vitesse (m/s)')
            plt.title(f"Vitesse à t = {i*dt:.2f} s")
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.pause(0.1)

    plt.show()
    
# def check_stability(nu, dx, dy, dt):
#     stability_limit = 1 / (2 * nu * (1 / dx**2 + 1 / dy**2))
#     print(f"Condition de stabilité : dt < {stability_limit:.4e} s")
#     if dt < stability_limit:
#         print(f"✅ Stable : dt = {dt:.4e} s < {stability_limit:.4e} s")
#     else:
#         print(f"⚠️ Instable : dt = {dt:.4e} s > {stability_limit:.4e} s")

# check_stability(nu, dx, dy, dt)

# if showProfileVitesse:
#     # affiche du profil de vitesse selon Y
#     fig = plt.figure(figsize=(11,7), dpi=100)
#     for i in tqdm.tqdm(range(6)):
#         nu = 1*10**(-6 + i)
#         u, v, p, stepcount = simulation_navier_stokes()
#         # x0 = (x.min()+ x.max()) //2
#         # x0 = np.argmin(np.abs(x - (x.min() + x.max())/2))
#         x0 = np.random.randint(0, len(x)-1)
#         V_x0 = u[:, int(x0)]
#         Re = max(V_x0)*1.5e-2/nu
#         plt.plot(V_x0, y,"--", label=f"nu={nu:.0e}\n$R_e = ${Re:.2e}", )
#     plt.xlabel("Vitesse selon $\\vec u_y$ (m/s)")
#     plt.ylabel("Y (cm)")
#     plt.legend(loc="center left")
    
simulation()
        