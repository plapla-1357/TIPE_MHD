import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramètres de simulation
nx, ny = 41, 41  # Nombre de points en x et y
dl = 1.0 / (nx - 1)  # Espacement entre les points
dt = 0.001  # Pas de temps
nu = 0.1  # Viscosité cinématique
nt = 5000  # Nombre d'itérations
rho = 1.0  # Densité
epsilon = 1e-8  # Petite valeur pour éviter les divisions par zéro

# Paramètres électromagnétiques
I = 3.0  # Courant en Ampères
L = 0.09  # Largeur en mètres
h = 0.003  # Hauteur en mètres
Bz = 1.0  # Champ magnétique en Tesla
J_y = I / (L * h)  # Densité de courant
F_x = J_y * Bz  # Force de Lorentz

# Initialisation des matrices
tau = np.zeros((nx, ny))  # Terme source de pression
u_x = np.zeros((nx, ny))  # Vitesse en x
u_y = np.zeros((nx, ny))  # Vitesse en y
p = np.zeros((nx, ny))  # Pression

# Fonction de mise à jour de la vitesse et pression
def solve_navier_stokes(nt, u, v, p, tau):
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        # Calcul du terme source de la pression (équation de Poisson)
        dudx = (un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dl + epsilon)
        dvdy = (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dl + epsilon)
        tau[1:-1, 1:-1] = (1 / dt) * (dudx + dvdy)
        
        # Éviter les valeurs extrêmes
        tau = np.clip(tau, -1e5, 1e5)
        
        # Résolution de l'équation de Poisson pour la pression
        for _ in range(50):  # Méthode itérative
            p[1:-1, 1:-1] = (
                (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - tau[1:-1, 1:-1] * dl**2) / 4
            )
            p[:, -1] = p[:, -2]  # Condition Neumann (dP/dx = 0)
            p[0, :] = p[1, :]
            p[:, 0] = p[:, 1]
            p[-1, :] = 0  # Condition Dirichlet (pression fixée)
        
        # Mise à jour des vitesses avec l'équation de Navier-Stokes
        u[1:-1, 1:-1] = np.clip(
            un[1:-1, 1:-1] - dt * (
                un[1:-1, 1:-1] * dudx + vn[1:-1, 1:-1] * dvdy
            ) - dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dl + epsilon)
            + nu * dt * ((un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dl**2 + (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dl**2)
            + dt * F_x / rho,
            -1e5, 1e5  # Éviter valeurs extrêmes
        )
        
        v[1:-1, 1:-1] = np.clip(
            vn[1:-1, 1:-1] - dt * (
                un[1:-1, 1:-1] * (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dl + epsilon) +
                vn[1:-1, 1:-1] * (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dl + epsilon)
            ) - dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dl + epsilon)
            + nu * dt * ((vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dl**2 + (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dl**2),
            -1e5, 1e5
        )
        
        # Conditions aux limites : Parois fixes
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0
        v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0
    
    return u, v, p

# Résolution des équations
u_x, nu_y, p = solve_navier_stokes(nt, u_x, u_y, p, tau)

# Affichage du champ de vitesse
X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
plt.quiver(X, Y, u_x, u_y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Champ de vitesse avec force de Lorentz")
plt.show()
