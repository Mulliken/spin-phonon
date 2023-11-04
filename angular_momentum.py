import numpy as np
from qutip import *
import matplotlib.pyplot as plt

# define three oscillators and frequencies, assuming same mass
N = 3
ax = tensor(destroy(N), qeye(N), qeye(N))
ay = tensor(qeye(N), destroy(N), qeye(N))
az = tensor(qeye(N), qeye(N), destroy(N))
wx = 2
wy = 2
wz = 2

# define coordinates and momenta, setting hbar = 1
x = (ax + ax.dag())/np.sqrt(2*wx)
y = (ay + ay.dag())/np.sqrt(2*wy)
z = (az + az.dag())/np.sqrt(2*wz)
px = -1j*(ax - ax.dag())/np.sqrt(wx/2)
py = -1j*(ay - ay.dag())/np.sqrt(wy/2)
pz = -1j*(az - az.dag())/np.sqrt(wz/2)

# define angular momentum operators
lx = y*pz - z*py
ly = z*px - x*pz
lz = x*py - y*px

# check if the angular momentum operators are correct
a_p = -1/np.sqrt(2) * 1j * (ax-1j*ay)
a_m = -1/np.sqrt(2) * 1j * (ax+1j*ay)
_lz = a_p.dag()*a_p - a_m.dag()*a_m
print("Lz is correct?", np.allclose(lz, _lz))
print(lz)
print(_lz)

# calculate the eigenstates of the angular momentum operators
ez, vz = lz.eigenstates()
print("Eigenvalues of Lz:", ez)
e, v = (lx * lx + ly * ly + lz * lz).eigenstates()
print("Eigenvalues of L^2:", e)

# calculate the lx, ly, lz in the eigenstates of l^2
lx_eigen = []
ly_eigen = []
lz_eigen = []
for i in range(len(e)):
    lx_eigen.append(expect(lx, vz[i]))
    ly_eigen.append(expect(ly, vz[i]))
    lz_eigen.append(expect(lz, vz[i]))
print("Lx:", lx_eigen)
print("Ly:", ly_eigen)
print("Lz:", lz_eigen)