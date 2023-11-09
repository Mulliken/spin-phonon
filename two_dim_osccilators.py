from qutip import *
import numpy as np

# define 2D isotropic harmonic oscillators
N = 4
omega = 1
a = tensor(destroy(N), qeye(N))
b = tensor(qeye(N), destroy(N))
x = (a + a.dag()) / np.sqrt(2)
y = (b + b.dag()) / np.sqrt(2)
H = omega * (a.dag() * a + b.dag() * b + 1)

# define the angular momentum operators
lz = 1j * (a * b.dag() - a.dag() * b)

# define the angular momentum annihilation operator
am = (a - 1j * b) / np.sqrt(2)
ap = (a + 1j * b) / np.sqrt(2)

(lz_val, e_val), v = simdiag([lz, H])

print("H  Eigenvalues:", e_val)
print("Lz Eigenvalues:", lz_val)

# initial state is a coherent state
psi0 = tensor(coherent(N, np.exp(1j*np.pi/2)), coherent(N, 1)).unit()
# psi0 = v[0]
# evolve the state and calculate the expectation values
tlist = np.linspace(0, 10, 100)
e_ops = [a.dag() * a, b.dag() * b, lz, x, y]
# print types of e_ops
result = sesolve(H, psi0, tlist, e_ops)

# plot the results
import matplotlib.pyplot as plt
# plt.plot(tlist, result.expect[0], label="n1")
# plt.plot(tlist, result.expect[1], label="n2")
# plt.plot(tlist, result.expect[2], label="lz")
# plt.plot(tlist, result.expect[3], label="x")
# plt.plot(tlist, result.expect[4], label="y")
# plt.legend()
# plt.ylim(-4, 4)
# plt.show()

plt.clf()
# draw the trajectory of the oscillator on the 2D plane
plt.plot(result.expect[3], result.expect[4])
plt.show()
