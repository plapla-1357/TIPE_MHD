from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


nx = 41
ny = 41
nt = 10
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(-1, 8, ny)
y = np.linspace(0.5, 2, nx)
X, Y = np.meshgrid(x, y)

# Réinitialisation des conditions initiales
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

##physical variables
rho = 1
nu = .1
I = 3
S = 1

# B = np.zeros_like(X)
# for i in range(len(x)):
#     for j in range(len(y)):
#         B[i, j] = get_champs_mag6(y[j], x[i])

B = np.ones_like(X)*15e-3





Fy = I/S*B
# Fy = 1
Fx = 0
dt = .01


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


velocity_magnitude_list = []

nsteps = 200  # Nombre d'étapes à visualiser

for step in range(nsteps):
#     un = u.copy()
#     vn = v.copy()

#     b = build_up_b(rho, dt, dx, dy, u, v)
#     p = pressure_poisson_periodic(p, dx, dy)

#     u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
#                     un[1:-1, 1:-1] * dt / dx * 
#                     (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
#                     vn[1:-1, 1:-1] * dt / dy * 
#                     (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
#                     dt / (2 * rho * dx) * 
#                     (p[1:-1, 2:] - p[1:-1, 0:-2]) +
#                     nu * (dt / dx**2 * 
#                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
#                     dt / dy**2 * 
#                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
#                     Fy[1: -1, 1: -1] * dt)

#     v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
#                     un[1:-1, 1:-1] * dt / dx * 
#                     (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
#                     vn[1:-1, 1:-1] * dt / dy * 
#                     (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
#                     dt / (2 * rho * dy) * 
#                     (p[2:, 1:-1] - p[0:-2, 1:-1]) +
#                     nu * (dt / dx**2 *
#                     (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
#                     dt / dy**2 * 
#                     (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])) + Fx * dt)

#     # Periodic BC u @ x = 2     
#     u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
#                 (un[1:-1, -1] - un[1:-1, -2]) -
#                 vn[1:-1, -1] * dt / dy * 
#                 (un[1:-1, -1] - un[0:-2, -1]) -
#                 dt / (2 * rho * dx) *
#                 (p[1:-1, 0] - p[1:-1, -2]) + 
#                 nu * (dt / dx**2 * 
#                 (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
#                 dt / dy**2 * 
#                 (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + Fy[1: -1, -1] * dt)

#     # Periodic BC u @ x = 0
#     u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
#                 (un[1:-1, 0] - un[1:-1, -1]) -
#                 vn[1:-1, 0] * dt / dy * 
#                 (un[1:-1, 0] - un[0:-2, 0]) - 
#                 dt / (2 * rho * dx) * 
#                 (p[1:-1, 1] - p[1:-1, -1]) + 
#                 nu * (dt / dx**2 * 
#                 (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
#                 dt / dy**2 *
#                 (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + Fy[1: -1, 0] * dt)

#     # Periodic BC v @ x = 2
#     v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
#                 (vn[1:-1, -1] - vn[1:-1, -2]) - 
#                 vn[1:-1, -1] * dt / dy *
#                 (vn[1:-1, -1] - vn[0:-2, -1]) -
#                 dt / (2 * rho * dy) * 
#                 (p[2:, -1] - p[0:-2, -1]) +
#                 nu * (dt / dx**2 *
#                 (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
#                 dt / dy**2 *
#                 (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])) + Fx * dt)

#     # Periodic BC v @ x = 0
#     v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
#                 (vn[1:-1, 0] - vn[1:-1, -1]) -
#                 vn[1:-1, 0] * dt / dy *
#                 (vn[1:-1, 0] - vn[0:-2, 0]) -
#                 dt / (2 * rho * dy) * 
#                 (p[2:, 0] - p[0:-2, 0]) +
#                 nu * (dt / dx**2 * 
#                 (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
#                 dt / dy**2 * 
#                 (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])) + Fx * dt)


#     # Wall BC: u,v = 0 @ y = 0,2
#     u[0, :] = 0
#     u[-1, :] = 0
#     v[0, :] = 0
#     v[-1, :]=0

#     # Update u and v (cf. ton code précédent, inchangé ici pour clarté)
#     # ...
#     # ... (mets ici tout ton code de mise à jour de u et v)

#     # Stocker la norme du champ de vitesse pour l'animation
#     velocity_magnitude = np.sqrt(u**2 + v**2)
#     velocity_magnitude_list.append(velocity_magnitude.copy())
    
    # Préparer les composantes du champ de vecteurs pour l'animation
    u_list = []
    v_list = []

    # Rejouer la simulation pour stocker u et v à chaque étape
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    for step in range(nsteps):
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
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0
        u_list.append(u.copy())
        v_list.append(v.copy())

    # Préparer la figure pour l'animation du champ de vecteurs
fig2, ax2 = plt.subplots(figsize=(8, 4))
quiv = ax2.quiver(X, Y, u_list[0], v_list[0], scale=5)
ax2.set_title("Évolution du champ de vecteur (u, v)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

def animate_quiver(i):
    quiv.set_UVC(u_list[i], v_list[i])
    ax2.set_title(f"Évolution du champ de vecteur (u, v) (étape {i})")
    return [quiv]

anim2 = FuncAnimation(fig2, animate_quiver, frames=len(u_list), interval=100, blit=False)
plt.show()
# fig, ax = plt.subplots(figsize=(8, 4))
# cax = ax.imshow(velocity_magnitude_list[0], cmap='viridis', origin='lower',
#                 extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
# fig.colorbar(cax)
# ax.set_title("Évolution du champ de vitesse |u|")
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# def animate(i):
#     cax.set_array(velocity_magnitude_list[i])
#     ax.set_title(f"Évolution du champ de vitesse |u| (étape {i})")
#     return [cax]

# anim = FuncAnimation(fig, animate, frames=len(velocity_magnitude_list), interval=100, blit=False)
# plt.show()