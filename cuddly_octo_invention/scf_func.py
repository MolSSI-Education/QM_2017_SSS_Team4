import numpy as np
import psi4

def mol(zmat):
    """
    Return a melecular geometry
    """
    return psi4.geometry(zmat)


def basis_set(mol, basis):
    """
    """
    return psi4.core.BasisSet.build(mol, target=basis)


def mints_helper(basis):
    """
    Make mintshelper from basis set
    """
    
    return psi4.core.MintsHelper(basis)


def A_diag(mints):
    """
    Calculate diagonalization term from ao_overlap integrals
    """
    
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    return np.array(A)

def diag(F, A):
    """
    Calculate the eigen values & functions of Fock Matrix
    """
    Fp = A.T.dot(F).dot(A)
    eps, Cp = np.linalg.eigh(Fp)
    C = A.dot(Cp)
    return eps, C


def exc_col(g, D):
    """
    Calculate the exchange and coulomb matrix
    """
    
    J = np.einsum("pqrs, rs->pq", g, D)
    K = np.einsum("prqs, rs->pq", g, D)
    return J, K


def Energy(E_old, F, H, D, mol):
    """
    Calculate energy and difference between new energy and old energy
    """
    
    E_electric = np.sum((F + H) * D)
    E_total = E_electric + mol.nuclear_repulsion_energy()
    E_diff = E_total - E_old
    return E_total, E_diff


def diis_e(F, D, S, A):
    """
    Calculate DIIS error matrix
    """

    diis_e = np.einsum('ij, jk, kl->il', F, D, S) - np.einsum('ij, jk, kl->il', S, D, F)
    diis_e = A.T.dot(diis_e).dot(A)
    return diis_e


def B_matrix(diis_count, diis_error):
    """
    Calculate the inner product matrix of error matrix
    """

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
    return B


def c_vec(B, diis_count):
    """
    Calculate the coeficients for the diis calculations
    """

    resid = np.zeros(diis_count + 1)
    resid[-1] = -1
    return np.linalg.solve(B, resid)


def fock_matrix_diis(fock_list, ci, F):
    """
    Calculate the fock matrix through diis
    """
    
    for num, c in enumerate(ci[:-1]):
        F += c * fock_list[num]
    return F
