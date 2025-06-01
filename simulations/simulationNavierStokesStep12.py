import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tqdm
import pandas as pd
from scipy.interpolate import interp1d
plt.rcParams.update({'font.size': 18})


###################################################
# paramètres de la simulation 
###################################################
nx = 41
ny = 41
nt = 10
nit = 50
c = 1
dt = .001


## variable physique 
rho = 1000#kg.m-3
eta = 1e-3
nu = eta/rho  #viscosité cinematique        
I = 2 #A      courant apliqué
l = 9e-2 #m    longeur de l'electrode
h = 8e-3 #m    hauteur d'eau au niveau des electrodes
e = 1.5e-2 #m epaiseur entre les electrodes
S = l*h
B0 = 11.22e-2  # T champ magnétique constant
U = 3 #V
E0 = U/e  # V/m champ électrique constant (pour la modelisation 2)
E = np.ones((nx, ny))*E0 

## variables chimiques
m = 60 # g  masse de sel
M = 58.44 # g.mol-1  masse molaire du chlorure de sodium
V = 300e-6 # m3 volume de la solution
Cond_Na = 5.008e-3 #S.m^2.mol^-1
Cond_Cl = 7.631e-3 #S.m2.mol^-1       (S = omh-1)


useMesure = False  # True pour utiliser les mesures de champs mag, False pour un champ magnétique constant
# choix du graphique affiché
showVecVitesse = False
showSteam = False
showSpeed = False
showProfileVitesse = False
showacceleration = False
showInfluanceI = False
showChampsMag = False
showCompareModels = False
showInfluance_e = True     #! Attention: 1min40 d'execution
showInfluance_e_Econst = False
showInfluance_m = False

# choix du modèle de force appliquée
model = 1
# 1: modèle 1 : hypothèse j const avec j = I/S
# 2: modèle 2 : hypothèse: loi d'ohm pour un fluide en mouvement j = Cond*(E - u*B)
##########################################################

######## consequences (variables deduites) ########

if useMesure:
    xs = -1e-2
    ys = 0.5e-2
else: 
    xs = 0
    ys = 0

dx = l / (nx - 1)
dy = e / (ny - 1)
x = np.linspace(xs, xs+l, ny)
y = np.linspace(ys, ys+e, nx)
X, Y = np.meshgrid(x, y)

def getConductivity(m):
    C = (m / M) / V  # mol.m-3 concentration de la solution 
    Cond = C * (Cond_Na + Cond_Cl)  # S.m-1 conductivité de la solution

    # # Données tabulées (à 25°C) : concentration en mol/L et conductivité en S/m
    # concentration_mol_L = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # conductivity_S_m = np.array([10.8, 17.2, 20.5, 21.9, 22.3, 21.8])
    # sigma_interp = interp1d(concentration_mol_L, conductivity_S_m, kind='cubic', fill_value="extrapolate")

    # def get_conductivity(concentration_mol_L):
    #     return float(sigma_interp(concentration_mol_L))

    # Cond = get_conductivity(C/ 1000)  # S/m conductivité de la solution (convertie de mol/L à mol/m3)
    return Cond
Cond = getConductivity(m)
I = Cond * E0 * S  # A courant appliqué (pour la modelisation 2)

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

def find_index(lst, value, tol=1e-8):
    for i, v in enumerate(lst):
        if abs(v - value) < tol:
            return i
    raise ValueError(f"{value} is not in list")

ChampMag["B"] = [b * 1e-2 for b in ChampMag["B"]]  # Conversion en Tesla
# ChampMag["X"] = [x * 1e-2 for x in ChampMag["X"]]  # Conversion en mètre
# ChampMag["Y"] = [y * 1e-2 for y in ChampMag["Y"]]  # Conversion en mètre
def get_real_champs_mag(x, y):
    return ChampMag["B"][find_index(ChampMag["X"],x) + find_index(ChampMag["Y"], y)]



# interpolation du champs magnétique 
def get_champs_mag6(x, y):
    x = x*1e2
    y = y*1e2
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

if useMesure:
    B = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            B[i, j] = get_champs_mag6(y[j], x[i])
    B = B.T
else:
    B = np.ones_like(X)*B0

# on en deduit la force appliquée
def get_Fy(u):
    if model == 1:  # modèle 1 : hypothèse j const avec j = I/S
        Fy = I/S*B /rho
    elif model == 2:  # modèle 2 : hypothèse: loi d'ohm pour un fluide en mouvement j = Cond*(E - u*B)
        Fy = Cond*(E - u*B)*B / rho
        # print(Fy)
    return Fy
Fx = 0


def build_up_b(rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # condition de periodicité en y = e
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # condition de periodicité en y = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    
    return b


def pressure_poisson_periodic(p, b, dx, dy):
    pn = np.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        #condition de periodicité en y = e
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # condition de periodicité en y = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # condition aux bords 
        p[-1, :] =p[-2, :]  # dp/dy = 0 a y = e
        p[0, :] = p[1, :]  # dp/dy = 0 a y = 0
    
    return p


def get_U_V(un, vn, p, u, v, dt, dx, dy, rho, nu):
    
    #cas de la matrique M_{n-2, n-2} au centre ou les condition au limite n'ont pas d'influance
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
                    get_Fy(un)[1: -1, 1: -1] * dt)

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

    # condition periodique droite     
    u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                (un[1:-1, -1] - un[1:-1, -2]) -
                vn[1:-1, -1] * dt / dy * 
                (un[1:-1, -1] - un[0:-2, -1]) -
                dt / (2 * rho * dx) *
                (p[1:-1, 0] - p[1:-1, -2]) + 
                nu * (dt / dx**2 * 
                (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                dt / dy**2 * 
                (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + get_Fy(un)[1: -1, -1] * dt)

    # condition periodique gauche
    u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                (un[1:-1, 0] - un[1:-1, -1]) -
                vn[1:-1, 0] * dt / dy * 
                (un[1:-1, 0] - un[0:-2, 0]) - 
                dt / (2 * rho * dx) * 
                (p[1:-1, 1] - p[1:-1, -1]) + 
                nu * (dt / dx**2 * 
                (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                dt / dy**2 *
                (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + get_Fy(un)[1: -1, 0] * dt)

    # condition periodique droite
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

    # condition periodique gauche
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


    # murs (vitesse nule)
    u[0, :] = 0
    u[-1, :] = 0
    v[0, :] = 0
    v[-1, :]=0
    return u, v


def simulation_navier_stokes():
    #initial conditions
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    b = np.zeros((ny, nx))
    udiff = 1
    stepcount = 0
    while udiff > .001: # detecte la convergence
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(p, b, dx, dy)

        u, v = get_U_V(un, vn, p, u, v, dt, dx, dy, rho, nu)
        
        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1
    return u, v, p, stepcount

def mesure_acceleration(nt):
    #conditions initial 
    u = np.zeros((ny, nx))
    un = np.zeros((ny, nx))

    v = np.zeros((ny, nx))
    vn = np.zeros((ny, nx))

    p = np.ones((ny, nx))
    pn = np.ones((ny, nx))

    b = np.zeros((ny, nx))
    udiff = 1
    stepcount = 0
    ax = []
    ay = []
    for i in tqdm.tqdm(range(nt)):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(p, b, dx, dy)

        u, v = get_U_V(un, vn, p, u, v, dt, dx, dy, rho, nu)
        ax.append(((u - un) / dt)[int(ny//2), int(ny//2)])
        ay.append(((v - vn) / dt)[int(ny//2), int(ny//2)])
    return ax, ay

def getA0():
    return mesure_acceleration(3)[0][2]

def getMeanSpeed(V_x0, e):
    return (1/e) * sum(np.array(V_x0)*dy) 


# chanps de vecteurs vitesses
if showVecVitesse:
    u, v, p, stepcount = simulation_navier_stokes()
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.pcolormesh(X, Y, p, cmap= "viridis", shading="auto", alpha=0.5)
    plt.colorbar(label='p (Pa)')
    plt.contour(X, Y, p, cmap=cm.viridis)  
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    plt.xlabel('Y (m)')
    plt.ylabel('X (m)')

if showSteam:
    u, v, p, stepcount = simulation_navier_stokes()
    # fig = plt.figure(figsize=(11,7), dpi=100)
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.streamplot(X, Y, u, v)
    plt.pcolormesh(X, Y, p, alpha=0.5, cmap="viridis", shading="auto")
    plt.colorbar(label="p (Pa)")
    plt.contour(X, Y, p, cmap=cm.viridis)
    plt.xlabel('Y (m)')
    plt.ylabel('X (m)')
    plt.title("Streamlines of velocity field")
    
if showSpeed:# affiche la vitesse selon Y
    u, v, p, stepcount = simulation_navier_stokes()
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.pcolormesh(X, Y, u, cmap="plasma", shading="auto")
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="u (m/s)")
    plt.title("Vitesse u en fonction de la position")
    
if showProfileVitesse:
    # affiche du profil de vitesse selon Y
    fig = plt.figure(figsize=(11,7), dpi=100)
    for i in tqdm.tqdm(range(2)):
        nu = 1*10**(-6 + i)
        u, v, p, stepcount = simulation_navier_stokes()
        x0 = (len(x)-1)//2
        V_x0 = u[:, int(x0)]
        Re = max(V_x0)*1.5e-2/nu
        plt.plot(V_x0, y,"--", label=f"nu={nu:.0e}\n$R_e = ${Re:.2e}", )
    plt.xlabel("Vitesse selon $\\vec u_y$ (m/s)")
    plt.ylabel("Y (m)")
    plt.legend(loc="center left")
        
if showacceleration:
    nt = 5000
    ax, ay = mesure_acceleration(nt)
    fig = plt.figure(figsize=(11,7), dpi=100) 
    t = np.linspace(0, nt*dt, nt)   
    plt.plot(t, ax, label="accerleration :$a_y$")
    plt.xlabel("Temps (s)")
    plt.ylabel("Accélération (m/s²)")
    plt.title("Accélération du fluide")
    plt.legend()

if showInfluanceI:
    Imin = 0.1
    Imax = 3.5
    N = 100
    Ilist = np.linspace(Imin, Imax, N)
    A0list = []
    for i in tqdm.tqdm(Ilist):
        I = i
        Fy = I/S*B /rho
        A0list.append(getA0())
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.plot(Ilist, A0list, "+",  label="A0")
    a, b = np.polyfit(Ilist, A0list, 1)
    # Affichage du coefficient a en notation scientifique LaTeX
    a_exp = int(np.floor(np.log10(abs(a)))) if a != 0 else 0
    a_mant = a / 10**a_exp if a != 0 else 0
    plt.plot(Ilist, a*Ilist + b, label=fr"Régression linéaire : $A_0 = {a_mant:.2f} \times 10^{{{a_exp}}} I + {b:.2f}$")
    plt.xlabel("Intensité (A)")
    plt.ylabel("Accélération (m/s²)")
    plt.legend()
    plt.grid()
    plt.title("Influence de l'intensité sur l'accélération du fluide")
    
if showChampsMag:
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.pcolormesh(X, Y, B, cmap="plasma", shading="auto")
    plt.colorbar(label="Champ magnétique (T)")
    plt.xlabel("Y (m)")
    plt.ylabel("X (m)")
    plt.title("Champ magnétique")
    
if showCompareModels:
    fig = plt.figure(figsize=(11,7), dpi=100)
    for i in range(2):
        model = i+ 1
        u, v, p, stepcount = simulation_navier_stokes()
        x0 = (len(x)-1)//2
        V_x0 = u[:, int(x0)]
        Re = max(V_x0)*1.5e-2/nu
        plt.plot(V_x0, y,"-"*(i+1), label=f"nu={nu:.0e}\n$R_e = ${Re:.2e}", )
    plt.xlabel("Vitesse selon $\\vec u_y$ (m/s)")
    plt.ylabel("Y (m)")
    plt.legend(loc="center left")
    
if showInfluance_m:
    fig = plt.figure(figsize=(11,7), dpi=100)
    mi = m # garde en memoire la valeur par defaut pour evité que ca cause des problemes
    meanSpeedList = []
    speedMax = []
    massList = []
    for m in tqdm.tqdm(range (10, 100, 10)):
        Cond = getConductivity(m)
        I = Cond * E0 * S  # A courant appliqué (pour la modelisation 2)
        u, v, p, stepcount = simulation_navier_stokes()
        x0 = (len(x)-1)//2
        V_x0 = u[:, int(x0)]
        Re = max(V_x0)*1.5e-2/nu
        plt.plot(V_x0, y,"--", label=f"m = {m} g", )
        plt.legend()
        plt.xlabel("Vitesse en ($m.s^{-1}$)")
        plt.ylabel("y (m)")
        massList.append(m)
        speedMax.append(max(V_x0))
        meanSpeedList.append(getMeanSpeed(V_x0, e))
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.plot(massList, meanSpeedList, "bo", label="Vitesse moyenne")
    a, b = np.polyfit(massList, meanSpeedList, 1)
    a_exp = int(np.floor(np.log10(abs(a)))) if a != 0 else 0
    a_mant = a / 10**a_exp if a != 0 else 0
    b_exp = int(np.floor(np.log10(abs(b)))) if b != 0 else 0
    b_mant = b / 10**b_exp if b != 0 else 0
    plt.plot(massList, a*np.array(massList) + b, "b--", label=fr"$V_m = {a_mant:.2f}\cdot 10^{{{a_exp}}} \times m + {b_mant:.2f} \cdot 10^{{{b_exp}}}$")
    plt.plot(massList, speedMax, "go", label="Vitesse maximale")
    c, d = np.polyfit(massList, speedMax, 1)
    c_exp = int(np.floor(np.log10(abs(c)))) if c != 0 else 0
    c_mant = c / 10**c_exp if c != 0 else 0
    d_exp = int(np.floor(np.log10(abs(d)))) if d != 0 else 0
    d_mant = d / 10**d_exp if d != 0 else 0
    plt.plot(massList, c*np.array(massList) + d, "g--", label=fr"$V_{{max}} = {c_mant:.2f} \cdot 10^{{{c_exp}}} \times m + {d_mant:.2f} \cdot 10^{{{d_exp}}}$")
    plt.xlabel("Masse (g)")
    plt.ylabel("Vitesse (m/s)")
    plt.title(f"Vitesse en fonction de la masse de soluté\n e = {e*1e2:.1f}cm E = {E0} V/m")
    plt.legend()
    plt.grid()
    
    
        
        
        
    m = mi
    
if showInfluance_e:
    fig = plt.figure(figsize=(11,7), dpi=100)
    meanSpeedList = []
    speedMax = []
    epaiseur = []
    for i in tqdm.tqdm(range (1, 41, 1)):
        e = (2+ i/10)*1e-3
        E0 = U/e
        I = Cond * E0 * S  # A courant appliqué (pour la modelisation 2)
        Cond = getConductivity(m)
        dx = l / (nx - 1)
        dy = e / (ny - 1)
        x = np.linspace(xs, xs+l, ny)
        y = np.linspace(ys, ys+e, nx)
        X, Y = np.meshgrid(x, y)
        u, v, p, stepcount = simulation_navier_stokes()
        x0 = (len(x)-1)//2
        V_x0 = u[:, int(x0)]
        Re = max(V_x0)*1.5e-2/nu
        plt.plot(V_x0, y,"--", label=f"e={e*1e3:.1f} mm\n$R_e = ${Re:.2e}", )
        plt.legend()
        plt.xlabel("Vitesse en ($m.s^{-1}$)")
        plt.ylabel("y (m)")
        plt.title("Vitesse selon Y pour différentes épaisseurs entre les électrodes")
        meanSpeedList.append(getMeanSpeed(V_x0, e))
        speedMax.append(max(V_x0))
        epaiseur.append(e)
    
    vmax = max(meanSpeedList)
    print(f"Vitesse moyenne maximale : {vmax:.4f} m/s")
    # Affichage sur le graphe
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.axhline(y=vmax, color='r', linestyle='--', label=f"$V_{{m, max}} = {vmax:.4f} m.s^{{-1}}$ \n e = {epaiseur[np.argmax(meanSpeedList)]*1e3:.1f} mm")
    plt.axvline(x=epaiseur[np.argmax(meanSpeedList)], color='r', linestyle='--')
    plt.plot(epaiseur, speedMax, "o-", label="Vitesse maximale")
    plt.plot(epaiseur, meanSpeedList, "o-", label="Vitesse moyenne")
    plt.xlabel("Épaisseur entre les électrodes (m)")
    plt.ylabel("Vitesse (m/s)")
    plt.title(f"Vitesse en fonction de l'éccartement entre les électrodes\n m = {m} g, U = {U} V")
    plt.legend()
    plt.grid()
if showInfluance_e_Econst:
    fig = plt.figure(figsize=(11,7), dpi=100)
    meanSpeedList = []
    speedMax = []
    epaiseur = []
    E0 = 200
    for i in tqdm.tqdm(range (1, 21, 2)):
        e = (2+ i/5)*1e-3
        dx = l / (nx - 1)
        dy = e / (ny - 1)
        x = np.linspace(xs, xs+l, ny)
        y = np.linspace(ys, ys+e, nx)
        X, Y = np.meshgrid(x, y)
        u, v, p, stepcount = simulation_navier_stokes()
        x0 = (len(x)-1)//2
        V_x0 = u[:, int(x0)]
        Re = max(V_x0)*1.5e-2/nu
        plt.plot(V_x0, y,"--", label=f"e={e*1e3:.1f} mm\n$R_e = ${Re:.2e}", )
        plt.legend()
        plt.xlabel("Vitesse en ($m.s^{-1}$)")
        plt.ylabel("y (m)")
        plt.title("Vitesse selon Y pour différentes épaisseurs entre les électrodes")
        meanSpeedList.append(getMeanSpeed(V_x0, e))
        speedMax.append(max(V_x0))
        epaiseur.append(e)
    
    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.plot(1/np.array(epaiseur), meanSpeedList, "+")
    a, b = np.polyfit(1/np.array(epaiseur), meanSpeedList, 1)
    plt.plot(1/np.array(epaiseur), a*(1/np.array(epaiseur)) + b, label=fr"Régression linéaire : $V_m = {a:.2f} \times 1/e + {b:.2f}$")
    plt.plot(1/np.array(epaiseur), speedMax, "+")
    c, d = np.polyfit(1/np.array(epaiseur), speedMax, 1)
    plt.plot(1/np.array(epaiseur), c*(1/np.array(epaiseur)) + d, label=f"Régression linéaire : $V_{max} =" + f"{c:.2f}" +"$\times 1/e + $"+f"{d:.2f}$")
    plt.grid()
    plt.legend()
    plt.xlabel("1/Épaisseur entre les électrodes (m⁻¹)")
    plt.ylabel("Vitesse moyenne (m/s)")
    plt.title("Vitesse moyenne en fonction de l'inverse de l'épaisseur entre les électrodes")

    fig = plt.figure(figsize=(11,7), dpi=100)
    plt.plot(epaiseur, speedMax, "o-", label="Vitesse maximale")
    plt.plot(epaiseur, meanSpeedList, "o", label="Vitesse moyenne")
    a_exp = int(np.floor(np.log10(abs(a)))) if a != 0 else 0
    a_mant = a / 10**a_exp if a != 0 else 0
    b_exp = int(np.floor(np.log10(abs(b)))) if b != 0 else 0
    b_mant = b / 10**b_exp if b != 0 else 0
    plt.plot(epaiseur, a*(1/np.array(epaiseur)) + b,"--", label=fr"Régression: $V_m = {a_mant:.2f} \cdot 10^{{{a_exp}}} \times \frac {{1}} {{e}} + {b_mant:.2f} \cdot 10^{{{b_exp}}}$")
    # plt.plot(epaiseur, c*(1/np.array(epaiseur)) + d, label=fr"Régression linéaire : $V_{max} = {c:.2f} \times 1/e + {d:.2f}$")
    plt.xlabel("Épaisseur entre les électrodes (m)")
    plt.ylabel("Vitesse (m/s)")
    plt.title(f"Vitesse en fonction de l'éccartement entre les électrodes\n m = {m} g, E = {U} V/m")
    plt.legend()
    plt.grid()

plt.show()
        
