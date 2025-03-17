from re import L
import numpy as np

import matplotlib.pyplot as plt

# Paramètres
viscosity = 0.001  # Viscosité dynamique de l'eau en Pa.s
pressure_gradient = 100  # Gradient de pression en Pa/m
thicknesses = [0.005, 0.00725, 0.01, 5]  # Épaisseurs de la conduite en mètres
L = 0.1  # Longueur de la conduite en mètres

# Fonction pour calculer le nombre de Reynolds
def reynolds_number(density, velocity, characteristic_length, viscosity):
    return (density * velocity * characteristic_length) / viscosity

# Paramètres supplémentaires
density = 1000  # Densité de l'eau en kg/m^3
characteristic_length = 0.01  # Longueur caractéristique en mètres
velocity = 0.01  # Vitesse moyenne en m/s

# Calcul du nombre de Reynolds
reynolds = reynolds_number(density, velocity, characteristic_length, viscosity)
print(f"Nombre de Reynolds: {reynolds}")


# Fonction pour calculer la vitesse de Poiseuille
def poiseuille_velocity(y, thickness, viscosity, pressure_gradient):
    return (pressure_gradient / (4 * viscosity)*L) * (thickness**2 - y**2)

# Discrétisation de l'espace

# Calcul et tracé des profils de vitesse
plt.figure(figsize=(10, 6))
for thickness in thicknesses:
    y_values = np.linspace(-thickness, thickness, 1000)
    velocities = poiseuille_velocity(y_values, thickness, viscosity, pressure_gradient)
    plt.plot(y_values, velocities, label=f'Épaisseur = {thickness*100*2} cm')

plt.xlabel('Position y (m)')
plt.ylabel('Vitesse (m/s)')
plt.title('Profil de vitesse de Poiseuille pour de l\'eau dans une conduite')
plt.legend()
plt.grid(True)
plt.show()