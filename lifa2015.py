import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


a = 1  # bond length
Kl, Kt = 1, 0.25  # spring constant
K = np.diag([Kl, Kt])
m1, m2 = 1, 1.2
a1 = np.array([np.sqrt(3), 0]) * a  # real lattice
a2 = np.array([np.sqrt(3)/2, 3/2]) * a
b1 = np.array([0, 4*np.pi/3]) / a  # reciprocal lattice
b2 = np.array([2*np.pi/np.sqrt(3), -2*np.pi/3]) / a
Mp = 1/2 * (b1 + b2)  # M point
Kp1 = 1/3 * b1 + 2/3 * b2  # K point
Kp2 = 2/3 * b1 + 1/3 * b2  # K' point


def R(theta):  # rotation matrix
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


theta1, theta2, theta3 = np.pi/6, 5*np.pi/6, -np.pi/2
K1 = R(theta1) @ K @ R(-theta1)
K2 = R(theta2) @ K @ R(-theta2)
K3 = R(theta3) @ K @ R(-theta3)


def get_D(k, K1, K2, K3, a1, a2, m1, m2):
    # get dynamics matrix, k=(kx, ky)
    D = np.zeros((4, 4), dtype=complex)
    D[:2, :2] = 1/m1 * (K1 + K2 + K3)
    D[2:, 2:] = 1/m2 * (K1 + K2 + K3)
    D[:2, 2:] = -1 / np.sqrt(m1*m2) * (
        K1 + K2*np.exp(-1j*k@a1) + K3*np.exp(-1j*k@a2))
    D[2:, :2] = D[:2, 2:].T.conj()
    return D


# Define the path in k-space
N = 100
# K --> Gamma --> K'
k_ = np.concatenate((np.linspace(Kp1, (0, 0), N+1),
                     np.linspace((0, 0), Kp2, N+1)[1:]))

# Gamma --> M --> K --> Gamma
# k_ = np.concatenate((np.linspace((0, 0), Mp, N+1),
#                      np.linspace(Mp, Kp1, N+1)[1:],
#                      np.linspace(Kp1, (0, 0), N+1)[1:]))


# Compute the phonon dispersion relation along the k-path
omega_ = np.zeros((len(k_), 4))
for i, k in enumerate(k_):
    D = get_D(k, K1, K2, K3, a1, a2, m1, m2)
    s, v = la.eigh(D)
    omega_[i] = np.sqrt(np.abs(s))
    print(v)
    # eigvals = np.linalg.eigvalsh(D)
    # omega_[i] = np.sqrt(np.abs(eigvals)) * np.sign(eigvals)


# Plot the phonon dispersion relation
fig = plt.figure(figsize=(8, 5))
plt.plot(omega_)
plt.xticks([0, N, 2*N], [r'$K$', r'$\Gamma$', r'$K^{\prime}$', ])
# plt.xticks([0, N, 2*N, 3*N], [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
# plt.show()
plt.grid(True)
fig.savefig('lifa2015.png')