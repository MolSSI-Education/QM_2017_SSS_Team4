import numpy as np
import psi4
import scf_func as scf


# print only 4 digits
np.set_printoptions(precision=14, suppress=True)

# Initialize variables for energy convergence and conditional Fock dampening
E_old = 0.0

   # Build a molecule
   # psi4.geometry builds a xyz matrix from a z-matrix structure in this case
   # physicists water molecule
mol = scf.mol("""
                 O
                 H 1 1.1
                 H 1 1.1 2 104""")

nel = 5 # number of electronic orbitals (RHF)
e_conv = 1.e-14 # energy convergence threshold
d_conv = 1.e-8  # density convergence threshold
max_iter = 50
bas = scf.basis_set(mol, "sto-3g") # Build a basis

mints = scf.mints_helper(bas) # Build a (MintsHelper = Object to get integrals from)

psi4.set_options({"BASIS": "STO-3G",
                  "SCF_TYPE": "DF",
                  "E_CONVERGENCE": 1.e-8})
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
mints2 = psi4.core.MintsHelper(wfn.basisset())

orb = wfn.basisset()

aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other="sto-3g")
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

Q1s_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
Q1s_tilde = np.squeeze(Q1s_tilde)

metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-0.5, 1.e-14)
metric = np.squeeze(metric)

V = np.array(mints.ao_potential()) # Build potential
T = np.array(mints.ao_kinetic()) # Build kinetic

H = T + V # Core Hamiltonian

F = H
F = np.array(F)

nbf = mints.nbf() # Get number of basis functions

fock_list = []
diis_error = []

S = np.array(mints.ao_overlap()) # Overlap matrix (S)
g = np.array(mints.ao_eri()) # Get AO two electron integral

p_1s = np.einsum("pq, pls->pls", metric, Q1s_tilde)
#p_1si = np.linalg.inv(p_1s)
#s_p = g.dot(p_1si)
A = scf.A_diag(mints)

eps, C = scf.diag(H, A) # Get eigenvalues eps, and eigenfunctions C
Cocc = C[:, :nel]
D = Cocc @ Cocc.T # Calculate density

for iteration in range(1, max_iter + 1):

    Chi = np.einsum("pls, ls->p", p_1s, D)
    J = np.einsum("lsp,p->ls", p_1s.transpose(1,2,0),Chi)
    zeta1 = np.einsum("lnP,pm->Pnp", p_1s.transpose(1,2,0),C[:,:nel])
    zeta2 = np.einsum("Pnl,pm->Pnp", p_1s,C[:,:nel])
    K = np.einsum("Pmp,Pnp->mn", zeta1, zeta2)

    # Get coulomb (J) and exchange (K) matrices
    # J, K = scf.exc_col(g, D)
    F = H + 2.0 * J - K # Calculate new Fock matrix

    # DIIS error build
    diis_error.append((A.T.dot((F.dot(D).dot(S)) - (S.dot(D).dot(F))).dot(A)))

    fock_list.append(F)

    # Build the AO gradient
    grad = F @ D @ S - S @ D @ F
    grad_rms = np.mean(grad ** 2) ** 0.5

    # Build the energy
    E_old, E_diff = scf.Energy(E_old, F, H, D, mol)

    if (iteration >= 2):
        diis_count = len(fock_list)
        ci = scf.c_vec(scf.B_matrix(diis_count, diis_error), diis_count)
        F = np.zeros_like(F)
        F = scf.fock_matrix_diis(fock_list, ci, F)

    Fp = A.T.dot(F).dot(A)
    eps, C = scf.diag(F, A) # Get eigenvalues eps, and eigenfunctions C
    Cocc = C[:, :nel]
    D = Cocc.dot(Cocc.T) # Calculate density
    print("Iter=%3d % 16.12f  % 8.4e % 8.4e "
          % (iteration, E_old, E_diff, grad_rms))

    if (abs(E_diff) < e_conv):
        break

    # Break if e_conv and d_conv are met
psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_old))
