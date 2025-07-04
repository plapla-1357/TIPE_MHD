from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

concentration_mol_L = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) # saturation a 6mol.L-1
conductivity_S_m = np.array([10.8, 17.2, 20.5, 21.9, 22.3, 21.8])
# plt.plot(concentration_mol_L, conductivity_S_m, 'ro', label='données expérimentales')

M = 58.44 # g.mol-1  masse molaire du chlorure de sodium
V = 300e-6 
def getConductivity(m):
    C = (m / M) / V  # mol.m-3 concentration de la solution 

    # Données tabulées (à 25°C) : concentration en mol/L et conductivité en S/m
    sigma_interp = interp1d(concentration_mol_L, conductivity_S_m, kind='cubic', fill_value="extrapolate")

    def get_conductivity(concentration_mol_L):
        return float(sigma_interp(concentration_mol_L))

    Cond = get_conductivity(C/ 1000)  # S/m conductivité de la solution (convertie de mol/L à mol/m3)
    return Cond


#conductivité eau de paris
cond_eau_paris = 5.49e-2 # S.m-1
S = 9e-2*8e-3 #m2
T = 20 #°C
e = 1.5e-2 # m
X = []
Y = []
for m in range(0, 100):
    M = 58.44 # g.mol-1  masse molaire du chlorure de sodium
    V = 300e-6 # m3 volume de la solution
    C = (m / M) / V  # mol.m-3 concentration de la solution 
    Cond_Na = 5.008e-3 #S.m^2.mol^-1
    Cond_Cl = 7.631e-3 #S.m2.mol^-1       (S = omh-1)
    Cond = C * (Cond_Na + Cond_Cl) # S.m-1 conductivité de la solution
    real_Cond = Cond*(1 + 0.02*(T-25))
    X.append(m)
    Y.append(real_Cond + cond_eau_paris) # correction de la conductivité de l'eau de paris
a, b = np.polyfit(X, Y, 1)
plt.plot(np.array(X)/M/0.3, Y, label=f'y = {a:.2e}x + {b:.2e}')
# plt.legend([f'y = {a:.2e}x + {b:.2e}'])
plt.title('Conductivité théorique en fonction de la masse de NaCl')
plt.xlabel('Concentration ($mol\cdot L^{-3}$)')
plt.ylabel('Conductivité ($S\cdot m^{-1}$)')
plt.grid()

mass = np.linspace(0, 110, 110)  # masse en g
conductivity = [getConductivity(m) for m in mass]
plt.plot((mass/M)/0.3, np.array(conductivity), 'g', label='conductivité empirique publiée')

X = np.array([40, 60, 75, 100])
C = X / M /0.3  # mol.m-3 concentration de la solution
# Y = np.array([0.75688073, 0.96313364, 1.12068966, 1.41891892])
Y = np.array([0.75688073, 0.96313364, 1.12068966, 1.21891892])
Z = Y * e / S
print(Z)
# b, c = np.polyfit(C, Z, 1)
# plt.plot(C, b*C + c, label =f"{b: .2f}x + {c: .2f}")
plt.errorbar(C, Z, yerr=3, fmt='+', label='données expérimentales avec erreur', capsize=5, elinewidth=2, markeredgewidth=2)
# plt.plot(C, Z, 'o', label='données expérimentales')
plt.axvline(6.1, color='r', linestyle='--', label='saturation: 6.1 mol.L-1')
plt.legend()
plt.show()
