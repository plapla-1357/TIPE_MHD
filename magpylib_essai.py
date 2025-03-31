import magpylib as magpy
from magpylib.magnet import Cylinder
import numpy as np
import matplotlib.pyplot as plt

# Définition des aimants cylindriques
magnets = [
    Cylinder(magnetization=(0, 0, 10000), dimension=(1, 2)).move((2, 2.5, 0)),
    Cylinder(magnetization=(0, 0, 10000), dimension=(1, 2)).move((2, 4.75, 0)),
    Cylinder(magnetization=(0, 0, 10000), dimension=(1, 2)).move((2, 7, 0))
]

# Définition d'une grille régulière (20x20 points)
x_range = np.linspace(0, 3, 200)
y_range = np.linspace(-1, 8, 200)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z_grid = np.zeros_like(X_grid)  # Plan Z = 0

# Création des points sous forme de liste [(x1,y1,z1), (x2,y2,z2), ...]
grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()])

# Calcul du champ magnétique en chaque point (devrait renvoyer un (400,3))
B_field = magpy.getB(magnets, grid_points)
print(B_field.shape)  # Affiche la forme du tableau de champs magnétiques
# Extraction de la composante Bz et reshaping en (20,20)
# On somme les contributions de chaque aimant
B_field = np.sum(B_field, axis=0)  # Devient (400,3)

# Extraction de la composante Bz et reshaping en (20,20)
Bz = B_field[:, 2].reshape(200, 200)

# Affichage avec imshow
plt.figure(figsize=(8, 6))
plt.imshow(Bz, extent=[0, 3, -1, 8], origin="lower", cmap="inferno", aspect="auto")
plt.colorbar(label="Champ Magnétique Bz (mT)")
plt.xlabel("Position X (m)")
plt.ylabel("Position Y (m)")
plt.title("Champ Magnétique Bz généré par 3 aimants cylindriques")
plt.show()
