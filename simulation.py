import numpy as np
import matplotlib.pyplot as plt
import warnings


#l'objectif est de simuler l'ecoulement dans un moteur MHD
#on decompose le moteur en sections selon x et y
#on suppose que la section est invariante selon z

#hyppothese: Fluide incompressible, newtonien, conducteur, non magnetique



ChampMag = {
    "X": [1.5]*19 + [1]*19 + [2]*19 + [0.5]*19,
    "Y": [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8] * 4,
    "B": [
        0.6, 3.5, 7.7, 13.8, 17.1, 16.9, 15.2, 15.5, 17.1, 18.3, 17, 15.7, 17, 19, 18.9, 13.9, 8.2, 3.9, 1.8,
        0.6, 3.6, 8.5, 14.2, 17.7, 17.6, 16, 17, 18.8, 19.6, 18.1, 16.5, 17.3, 19, 18.1, 12.4, 7.4, 3.7, 1,
        0.2, 1.8, 4.8, 8.8, 11.1, 10.8, 9.6, 9.5, 10.3, 10.6, 10, 9.4, 10, 11.2, 11, 8.5, 4.5, 2.1, 0.6,
        0.8, 3.4, 8.1, 13.3, 15.8, 15.5, 14.1, 15, 17, 17.5, 15.9, 14.5, 15.6, 16.8, 16, 11.2, 5.9, 2.6, 0.7
    ]
}

def get_real_champs_mag(x, y):
    return ChampMag["B"][ChampMag["X"].index(x) + ChampMag["Y"].index(y)]

def get_champs_mag(x, y):
    # Trouver les indices des points les plus proches
    # print([abs(ChampMag["X"][i]-x) for i in range(len(ChampMag["X"]))])
    distances = [np.sqrt((ChampMag["X"][i] - x)**2 + (ChampMag["Y"][i] - y)**2) for i in range(len(ChampMag["X"]))]
    indices = [i for i in range(len(distances)) if distances[i] <= 1.5]
    
    if not indices:
        # warnings.warn("\033[94m One of the B values is None \033[0m")
        print("No indices found")
        return None
    
    # Calculer la moyenne pondérée des valeurs de B autour des indices trouvés
    B_values = [ChampMag["B"][i] for i in indices]
    weights = [1 / distances[i] if distances[i] != 0 else 1 for i in indices]
    return np.average(B_values, weights=weights)

def get_champs_mag2(x, y):
    if x in ChampMag["X"] and y in ChampMag["Y"]:
        return ChampMag["B"][ChampMag["X"].index(x) + ChampMag["Y"].index(y)]
    indices = [ChampMag["X"].index(((x*2)//1)/2), ChampMag["X"].index((((x+1)*2)//1)/2), ChampMag["Y"].index(((x*2)//1)/2), ChampMag["Y"].index((((x+1)*2)//1)/2)]
    B_values = [ChampMag["B"][i] for i in indices]
    distances = [x%1, y%1, 1-(x+1)%1, 1-(y+1)%1]
    return np.average(B_values, weights=distances)

def get_champs_mag3(x, y):
    # Trouver les quatre points les plus proches
    # if x in ChampMag["X"] and y in ChampMag["Y"]:
    #     return ChampMag["B"][ChampMag["X"].index(x) + ChampMag["Y"].index(y)]
    if x in ChampMag["X"]:
        x1, x2 = x, x+0.5
        y1, y2 = np.floor(y * 2) / 2, np.ceil(y * 2) / 2
    elif y in ChampMag["Y"]:
        x1, x2 = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
        y1, y2 = y, y+0.5
    else:
        x1, x2 = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
        y1, y2 = np.floor(y * 2) / 2, np.ceil(y * 2) / 2
    
    # Obtenir les valeurs de B aux quatre points
    B11 = get_champs_mag(x1, y1)
    B12 = get_champs_mag(x1, y2)
    B21 = get_champs_mag(x2, y1)
    B22 = get_champs_mag(x2, y2)
    
    # Si l'une des valeurs est None, retourner None
    if None in [B11, B12, B21, B22]:
        return None
    
    # Interpolation bilinéaire
    B = (B11 * (x2 - x) * (y2 - y)*4 +
         B21 * (x - x1) * (y2 - y)*4 +
         B12 * (x2 - x) * (y - y1)*4 +
         B22 * (x - x1) * (y - y1)*4)
    
    return B

def get_champs_mag4(x,y):
    # Trouver les quatre points les plus proches
    x1, x2 = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
    y1, y2 = np.floor(y * 2) / 2, np.ceil(y * 2) / 2
    
    # Obtenir les valeurs de B aux quatre points
    B11 = get_champs_mag(x1, y1)
    B12 = get_champs_mag(x1, y2)
    B21 = get_champs_mag(x2, y1)
    B22 = get_champs_mag(x2, y2)
    
    # Si l'une des valeurs est None, retourner None
    if None in [B11, B12, B21, B22]:
        warnings.warn("\033[94m One of the B values is None \033[0m")
        return None

    #interpolation 
    t1 = (x2 - x)/(x2 - x1)
    t2 = (y2 - y)/(y2 - y1)
    B = (1-t1)*(1-t2)*B11 + t1*(1-t2)*B21 + (1-t1)*t2*B12 + t1*t2*B22
    return B

def get_champs_mag5(x, y):
    if x in ChampMag["X"] and y in ChampMag["Y"]:
        return ChampMag["B"][ChampMag["X"].index(x) + ChampMag["Y"].index(y)]
    if x in ChampMag["X"]:
        y1, y2 = np.floor(y * 2) / 2, np.ceil(y * 2) / 2
        B1 = get_real_champs_mag(x, y1)
        B2 = get_real_champs_mag(x, y2)
        t = (y-y1)/(y2 - y1)
        B = B1*(1-t) + B2*t
        return B
    if y in ChampMag["Y"]:
        x1, x2 = np.floor(x * 2) / 2, np.ceil(x * 2) / 2
        B1 = get_real_champs_mag(x1, y)
        B2 = get_real_champs_mag(x2, y)
        t = (x-x1)/(x2 - x1)
        B = B1*(1-t) + B2*t
        return B
    #trouver les 3 points les plus proches
    if (2*x)%1 + (2*y)%1 < 1: #triange inferieur
        x1, x2, x3 = float(np.floor(x * 2)) / 2, float(np.ceil(x * 2) )/ 2, float(np.floor(x * 2)) / 2
        y1, y2, y3 = float(np.floor(y * 2)) / 2, float(np.floor(y * 2)) / 2,float( np.ceil(y * 2)) / 2
        
        B1 = get_real_champs_mag(x1, y1)
        B2 = get_real_champs_mag(x2, y2)
        B3 = get_real_champs_mag(x3, y3)
        
        if x1 == x2:         
            print(x1, x2)
        tx = (x - x1) / (x2 - x1)
        Bh2 = B1*(1-tx) + B2*tx
        Bh1 = B3*(1-tx) + B2*tx
        
        yH1 = 1-tx
        ty = (y - y1) / (yH1 - y1)
        B = Bh2*(1-ty) + Bh1*ty
        return B
    else: # triangle supperieur
        x1, x2, x3 = np.ceil(x * 2) / 2, np.ceil(x * 2) / 2, np.floor(x * 2) / 2
        y1, y2, y3 = np.ceil(y * 2) / 2, np.floor(y * 2) / 2, np.ceil(y * 2) / 2
        
        tx = (x - x1) / (x3 - x1)
        ty = (y - y1) / (y2 - y1)
        h = (tx, 1 - tx)
        
        B1 = get_real_champs_mag(x1, y1)
        B2 = get_real_champs_mag(x2, y2)
        B3 = get_real_champs_mag(x3, y3)
        
        Bh = B2*(1-tx) + B3*tx
        
        ty2 = (1-tx)*ty
        B = Bh*(ty2) + (B1*(1-tx) + (tx)*B3)*(1-ty2)
        return B


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



class cell:
    X = np.array([1,0,0])
    Y = np.array([0,1,0])
    Z = np.array([0,0,1])
    
    def __init__(self, x, y, Xi, Yi, B, V):
        self.x = x
        self.X_index = Xi
        self.y = y
        self.Y_index = Yi
        self.B = get_champs_mag3(x, y)
        self.v = np.array([0, 0, 0])
        self.a = np.array([0, 0, 0])
        self.p = 1e3 #kg/m^3
        self.h = h
        self.L = L
        self.l = l
        self.V = h*l*L #m^3
        self.m = self.p * self.V
        
    def update(self, j, dt):
        # force appliquee a l'eau: Force de Laplace, Force de viscosite, Force de gravite
        # self.a = 1/self.m * (self.B * self.j * self.V)*X 
        # self.v = self.v + self.a * dt
        pass 
print(get_champs_mag(1.5, 10))

def showHeatMap(): 
    X = np.linspace(0.5, 2, 1000)
    Y = np.linspace(-1, 8, 1000)
    plt.pcolormesh(X, Y, [[get_champs_mag6(x, y) for x in X] for y in Y], cmap="plasma")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Colormap of Magnetic Field B")
    plt.colorbar()
    plt.show()


def show3DMap():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Création d'une surface continue
    X = np.linspace(0.5, 2, 50)
    Y = np.linspace(-1, 8, 50)
    X_grid, Y_grid = np.meshgrid(X, Y)
    B_grid = np.zeros_like(X_grid)
    
    for i in range(len(Y)):
        for j in range(len(X)):
            B_grid[i, j] = get_champs_mag6(X[j], Y[i])
    
    ax.plot_surface(X_grid, Y_grid, B_grid, cmap='plasma', edgecolor='none')
    ax.set_title("Champ Magnétique Interpolé en fonction de la position (X, Y, B)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("B")
    
    plt.show()


if __name__ == "__main__":
    # showHeatMap()
    show3DMap()
    



