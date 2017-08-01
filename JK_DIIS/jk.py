import numpy as np
import psi4

def jk(mol,C,D):
    #np.set_printoptions(suppress=True, precision=4)
    nel = 5
    # Build a basis
    bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
    #bas.print_out()

    # Get orbital basis from a wavefunction object
    #orb = wfn.basisset()

    # Build the complementary JKFIT basis for the aug-cc-pVDZ basis
    aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other="aug-cc-pVDZ")
    #aux.print_out()

    # The zero basis set
    zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    #nbf = mints.nbf()

    #if (nbf > 100):
    #raise Exception("More than 100 basis functions!")

    # Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
    Qls_tilde = mints.ao_eri(zero_bas, aux, bas, bas)
    Qls_tilde = np.squeeze(Qls_tilde) # remove the 1-dimensions
    #print(Qls_tilde)

    # Build & invert Coulomb metric [J], dimension (1, Naux, 1, Naux)
    metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
    metric.power(-0.5, 1.e-14)
    metric = np.squeeze(metric) # remove the 1-dimensions
    #print(metric)

    # Build (P|\lambda\sigma) or Pls = [J^{-1/2}]_{PQ}\tilde{(Q|\lambda\sigma)} = metric_{PQ}'*'Qls_tilde
    Pls = np.einsum("pq,qls->pls", metric, Qls_tilde)


    # Build chi_p = Pls '*' Dls
    chi_p = np.einsum("pls,ls->p", Pls,D)

    # Build Coulomb matrix or J_ls = lsP '*'chi_P
    J= np.einsum("pls,p->ls", Pls, chi_p)

    # Build zeta1_Pmp = Pls.transpose_msP '*' C_ps
    #zeta1 = np.einsum("msP,ps->Pmp", Pls.transpose(1,2,0),C[:,:nel])
    zeta1 = np.einsum("Pls,si->Pli", Pls, C[:,:nel])
    # Build zeta2_Pnp = Pls_Pnl '*' C_pl
    # zeta2 = np.einsum("Pnl,pm->Pnp", Pls, C[:,:nel])

    # Build Exchange matrix or K_mn = zeta1_Pmp '*' zeta2_Pnp
    K = np.einsum("Ppi,Pqi->pq", zeta1, zeta1)

    return J, K
