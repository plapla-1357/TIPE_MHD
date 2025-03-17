import numpy as np
import matplotlib.pyplot as plt


#constantes
e = 1.9
# environement 
E = np.array([[15., 0., 0.]]) # champ electrique selon x
B = np.array([0., 0.5, 0.]) # champs magnétique selon y
q = e
m = 3.8 # Na

#variables de la simulation
n = 10000
Te = 0.1

T = np.array([i for i in range(n)])
V = np.zeros((n, 3))
X = np.zeros((n, 3))
print(V)
ax = plt.figure().add_subplot(projection='3d')
# ax = plt.figure().add_subplot()

def euler():
    A = np.array([[1, -Te*q*B[2]/m, Te*q*B[1]/m],
                  [Te*q*B[2]/m, 1, -Te*q*B[0]/m],
                  [-Te*q*B[1]/m, Te*q*B[0]/m, 1]]).T
    invA = np.linalg.inv(A)
    print(invA)
    for i in range(1, n):
        V[i] = np.dot(V[i-1], invA) + q*E
        # print(V[i].T.shape)
        # print((q*E).shape)
        # print("V[i-1]", V[i-1])
        # print((invA @ V[i-1]))
        # print((invA @ V[i-1]) + q*E)
        # print(V[i])
    for i in range(1, n):
        X[i] = X[i-1] + Te*V[i]
    
    print(V)
    # ax.plot(V.T[0], V.T[1], V.T[2], ".")
    ax.plot(X.T[0], X.T[1], X.T[2], ".")
    # ax.view_init(elev=20., azim=-35, roll=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    # plt.plot(T, V)
    # plt.show()
    
    
euler()
        
    

