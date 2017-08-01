import numpy as np
import psi4
import scf_func as scf

np.set_printoptions(suppress=True, precision=4)

mol = scf.mol("""
O
H 1 1.1
H 1 1.1 2 104
""")

# Build a molecule
mol.update_geometry()
mol.print_out()

e_conv = 1.e-6
d_conv = 1.e-6
nel = 5

# Build a basis
bas = scf.basis_set(mol, "sto-3g")

# Build a MintsHelper
mints = scf.mints_helper(bas)
nbf = mints.nbf()

if (nbf > 100):
    raise Exception("More than 100 basis functions!")

V = np.array(mints.ao_potential())
T = np.array(mints.ao_kinetic())

# Core Hamiltonian
H = T + V

S = np.array(mints.ao_overlap())
g = np.array(mints.ao_eri())

A = scf.A_diag(mints)

eps, C = scf.diag(H, A)
Cocc = C[:, :nel]
D = Cocc @ Cocc.T

E_old = 0.0
F = None
for iteration in range(25):
                        
    J, K = scf.exc_col(g, D)

    F = H + 2.0 * J - K

    # Build the AO gradient
    grad = F @ D @ S - S @ D @ F

    grad_rms = np.mean(grad ** 2) ** 0.5

    # Build the energy
    
    E_old, E_diff = scf.Energy(E_old, F, H, D, mol)

    print("Iter=%3d  E = % 16.12f  E_diff = % 8.4e  D_diff = % 8.4e" %
            (iteration, E_old, E_diff, grad_rms))

    # Break if e_conv and d_conv are met
    if (E_diff < e_conv) and (grad_rms < d_conv):
        break

    eps, C = scf.diag(F, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

print("SCF has finished!\n")

psi4.set_output_file("output.dat")
psi4.set_options({"scf_type": "pk"})
psi4_energy = psi4.energy("SCF/sto-3g", molecule=mol)
print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_old))
