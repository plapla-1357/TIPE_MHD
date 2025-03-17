import numpy as np
import matplotlib.pyplot as plt


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

def get_champs_mag(x, y):
    # Trouver les indices des points les plus proches
    # print([abs(ChampMag["X"][i]-x) for i in range(len(ChampMag["X"]))])
    distances = [np.sqrt((ChampMag["X"][i] - x)**2 + (ChampMag["Y"][i] - y)**2) for i in range(len(ChampMag["X"]))]
    indices = [i for i in range(len(distances)) if distances[i] <= 1.5]
    
    if not indices:
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
    if x in ChampMag["X"] and y in ChampMag["Y"]:
        return ChampMag["B"][ChampMag["X"].index(x) + ChampMag["Y"].index(y)]
    elif x in ChampMag["X"]:
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

# def simulation()

print(get_champs_mag(1.5, 10))

X = np.linspace(0.5, 2, 100)
Y = np.linspace(-1, 8, 100)
plt.pcolormesh(X, Y, [[get_champs_mag3(x, y) for x in X] for y in Y], cmap="plasma")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Colormap of Magnetic Field B")
plt.colorbar()
plt.show()
