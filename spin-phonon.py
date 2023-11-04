import numpy as np
from qutip import *

# Parameters
omega_e = 1  # Electron energy level splitting
omega_s = 0  # Spin energy
omegas = [1]*3  # Oscillator 1 frequency
gs = [0.]*3  # Coupling strength for oscillator 2
# g_soc = 0.1
g_ang_mom = [0.1]*3
gammas = [0.5]*3 # Damping rate for oscillators
N = 3         # Number of oscillator levels


# transformation matrix from phonons to coordinates
trans_mat = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                      [1/np.sqrt(2), -1/np.sqrt(2), 0],
                      [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)]])
print(trans_mat@trans_mat.T)
# trans_mat = np.eye(3)

# lx, ly, lz are the components of the spin-orbit coupling vector
lx = ly = 0.005
lz = 0.005

oscillators = 1
# Operators
# Electron Hilbert space
e_z = tensor(sigmaz(), qeye(2))
e_x = tensor(sigmax(), qeye(2))
e_p = tensor(sigmap(), qeye(2))
e_m = tensor(sigmam(), qeye(2))
# Spin Hilbert space
s_z = tensor(qeye(2), sigmaz())/2
s_x = tensor(qeye(2), sigmax())/2
s_y = tensor(qeye(2), sigmay())/2


def get_angular_momentum(anih_ops, trans_mat):
    x = sum([ci * (op + op.dag()) for ci, op in zip(trans_mat[0], anih_ops)])
    y = sum([ci * (op + op.dag()) for ci, op in zip(trans_mat[1], anih_ops)])
    z = sum([ci * (op + op.dag()) for ci, op in zip(trans_mat[2], anih_ops)])

    px = 1j * sum([ci * (op.dag() - op) for ci, op in zip(trans_mat[0], anih_ops)])
    py = 1j * sum([ci * (op.dag() - op) for ci, op in zip(trans_mat[1], anih_ops)])
    pz = 1j * sum([ci * (op.dag() - op) for ci, op in zip(trans_mat[2], anih_ops)])

    lx = y * pz - z * py
    ly = z * px - x * pz
    lz = x * py - y * px
    print("Coordinates is Hermitian?", x.isherm, y.isherm, z.isherm)
    print("Momentum is Hermitian?", px.isherm, py.isherm, pz.isherm)
    print("y*pz Hermitian?", (y*pz).isherm)
    print("Lx:", np.linalg.norm(lx), "isHermitian?", lx.isherm)
    print("Ly:", np.linalg.norm(ly), "isHermitian?", ly.isherm)
    print("Lz:", np.linalg.norm(lz), "isHermitian?", lx.isherm)

    return lx, ly, lz


# Oscillator Operators and collapse operators
if oscillators:
    e_z = tensor(e_z, qeye(N), qeye(N), qeye(N))
    e_x = tensor(e_x, qeye(N), qeye(N), qeye(N))
    e_p = tensor(e_p, qeye(N), qeye(N), qeye(N))
    e_m = tensor(e_m, qeye(N), qeye(N), qeye(N))
    s_z = tensor(s_z, qeye(N), qeye(N), qeye(N))
    s_x = tensor(s_x, qeye(N), qeye(N), qeye(N))
    s_y = tensor(s_y, qeye(N), qeye(N), qeye(N))
    
    a_1 = tensor(destroy(N), qeye(N), qeye(N))
    a_2 = tensor(qeye(N), destroy(N), qeye(N))
    a_3 = tensor(qeye(N), qeye(N), destroy(N))
    l_xyz = get_angular_momentum([a_1, a_2, a_3], trans_mat)
    ang_mom_state_y = l_xyz[1].eigenstates()[1][-1]
    ang_mom_state_x = l_xyz[0].eigenstates()[1][-1]
    ang_mom_state_z = l_xyz[2].eigenstates()[1][1]
    ang_mom_state = ang_mom_state_z.unit()

    a_ops = [tensor(qeye(2), qeye(2), op) for op in [a_1, a_2, a_3]]
    
    l_xyz = get_angular_momentum(a_ops, trans_mat)

    H_vib = sum(omega * op.dag() * op for omega, op in zip(omegas, a_ops))

    # print the commutator of lz and H_vib
    print(np.linalg.norm(commutator(H_vib, l_xyz[-1])))
    # exit()
    H_int = sum([g*e_z*(op+op.dag()) for g, op in zip(gs, a_ops)])
    # g_1 *  + g_2 * (e_p * a_2 + e_m * a_2.dag())
    H_ang_mom_coupling = sum(g * l * s for g,l,s in zip(g_ang_mom, l_xyz, [s_x,s_y,s_z]))
    # Collapse operators for Lindblad dissipation
    c_ops = [np.sqrt(gamma) * a for gamma, a in zip(gammas, a_ops)]
else:
    H_vib = 0
    H_int = 0
    H_ang_mom_coupling = 0
    c_ops = []

# Hamiltonian components
H_e = 0.5 * omega_e * e_z + 0.5 * omega_e * e_x
H_s = omega_s * s_z
H_soc = -1j * (e_p * (lx * s_x + ly*s_y + lz*s_z) -  e_m * (lx * s_x + ly*s_y + lz*s_z))
# H_soc_vib = -1j * (e_p * (lx * s_x + ly*s_y + lz*s_z) - e_m * (lx * s_x + ly*s_y + lz*s_z)) * (a_1 + a_1.dag() + a_2 + a_2.dag()) * g_soc
# H_spin_vib = -1j * (e_p * (lx * s_x + ly*s_y + lz*s_z) - e_m * (lx * s_x + ly*s_y + lz*s_z)) * (a_1 + a_1.dag() + a_2 + a_2.dag()) * g_soc
# Total Hamiltonian
H0 = H_e + H_s + H_vib + H_int + H_ang_mom_coupling
H1 = H_e + H_s + H_vib + H_int - H_ang_mom_coupling.dag()
# Initial state
vib_ground = basis(N, N-2)
vib_excited = basis(N, N-1)
state1 = (vib_ground + 1j*vib_excited).unit()
state2 = (vib_ground - 1j*vib_excited).unit()
spin_superposition = (basis(2,0) + basis(2,1)).unit()

if oscillators: 
    psi0 = tensor(basis(2,0), spin_superposition, ang_mom_state)
    psi1 = tensor(basis(2,0), basis(2,0), ang_mom_state)
else:
    psi0 = tensor(basis(2,0), basis(2,0))

# Time evolution
times = np.linspace(0, 25, 200)
result0 = mesolve(H0, psi0, times, c_ops)
result1 = mesolve(H1, psi1, times, c_ops)

# For visualization, let's plot the expectation value of sigma_z and the oscillator number operators
ez_exp0 = expect(e_z, result0.states)
sz_exp0 = expect(s_z, result0.states)
# sx_exp0 = expect(s_x, result0.states)
# sx_exp1 = expect(s_x, result1.states)
sz_exp1 = expect(s_z, result1.states)

if oscillators:
    n1_exp = expect(a_ops[0].dag()*a_ops[0], result0.states)
    n2_exp = expect(a_ops[1].dag()*a_ops[1], result0.states)

# print("Hamiltonian:", H)
print("Hamiltonain Hermitian?", H0.isherm, H1.isherm)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(times, sz_exp0, label='Spin $\\sigma_z 0$')
plt.plot(times, sz_exp1, label='Spin $\\sigma_z 1$')
# for i, l in enumerate(l_xyz):
    # plt.plot(times, expect(l, result0.states), label=f'Angular momentum {i}')
# plt.plot(times, expect(sum(l*l for l in l_xyz), result0.states), label='$L^2$')
# plt.plot(times, sx_exp0, label='Spin $\\sigma_x 0$')
# plt.plot(times, sx_exp1, label='Spin $\\sigma_x 1$')

# plt.plot(times, ez_exp0, label='Electron $\\sigma_z 0$')
# if oscillators:
#     plt.plot(times, n1_exp, label='Oscillator 1 population')
#     plt.plot(times, n2_exp, label='Oscillator 2 population')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Expectation value')
plt.savefig('spin-phonon.png')
