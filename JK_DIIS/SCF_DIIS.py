import numpy as np
import psi4
import jk

# print only 4 digits
np.set_printoptions(precision=14, suppress=True)
psi4.set_output_file("output.dat", True)


# Diagonalize Core H
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C

# Initialize variables for energy convergence and conditional Fock dampening
E_old = 0.0
F_old = None

# Build a molecule
# psi4.geometry builds a xyz matrix from a z-matrix structure in this case
# physicists water molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104""")

nel = 5 # number of electronic orbitals (RHF)
e_conv = 1.e-14 # energy convergence threshold
d_conv = 1.e-8  # density convergence threshold
damp_value = 0.20 
damp_start = 5 

bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ") # Build a basis

mints = psi4.core.MintsHelper(bas) # Build a (MintsHelper = Object to get integrals from)

V = np.array(mints.ao_potential()) # Build potential

T = np.array(mints.ao_kinetic()) # Build kinetic

H = T + V # Core Hamiltonian

F = H # Initial guess for Fock matrix is core Hamiltonian

F = np.array(F)
nbf = mints.nbf() # Get number of basis functions

# We want to limit the ammount of memory we call hard stop at 100 basis functions
if (nbf > 100):
    raise Exception("More than 100 basis functions!")

fock_list = []
diis_error = []

S = np.array(mints.ao_overlap()) # Overlap matrix (S)

g = np.array(mints.ao_eri()) # Get AO two electron integral

# S^-1/2
A = mints.ao_overlap()
A.power(-0.5, 1.e-14)
A = np.array(A)
eps, C = diag(F, A) # Get eigenvalues eps, and eigenfunctions C
Cocc = C[:, :nel]
D = Cocc @ Cocc.T # Calculate density

for iteration in range(40):

    # Get coulomb (J) and exchange (K) matrices
    #J = np.einsum("pqrs, rs -> pq", g, D)
    #K = np.einsum("prqs, rs -> pq", g, D)
    J, K = jk.jk(mol, C, D)
    F = H + 2.0 * J - K # Calculate new Fock matrix

    # DIIS error build
    diis_e = np.einsum('ij, jk, kl->il', F, D, S) - np.einsum('ij, jk, kl->il', S, D, F)
    diis_e = A.T.dot(diis_e).dot(A)
    fock_list.append(F)
    diis_error.append(diis_e)
    # Build the AO gradient       
    
    # Build the energy
    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()
    E_diff = E_total - E_old
    E_old = E_total

    if (iteration >= 1):
        diis_count = len(fock_list)
        B = np.empty((diis_count + 1, diis_count + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for num1, e1 in enumerate(diis_error):
            for num2, e2 in enumerate(diis_error):
                if num2 > num1: continue
                val = np.einsum('ij, ij->', e1, e2)
                B[num1, num2] = val
                B[num2, num1] = val

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        resid = np.zeros(diis_count + 1)
        resid[-1] = -1

        ci = np.linalg.solve(B, resid)

        F = np.zeros_like(F)
        for num, c in enumerate(ci[:-1]):
            F += c * fock_list[num]

    Fp = A.T.dot(F).dot(A)
    eps, C = diag(F, A) # Get eigenvalues eps, and eigenfunctions C
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T # Calculate density

    print("Iter=%3d % 16.12f  % 8.4e  "
          % (iteration, E_total, E_diff))
    if (abs(E_diff) < e_conv):
        break

    # Break if e_conv and d_conv are met
psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "df"})
psi4_energy = psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)
print("Total energy calculated: ", E_total)
print("Psi4 energy: ", psi4_energy)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))
