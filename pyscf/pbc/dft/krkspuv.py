#!/usr/bin/env python
#
# Prototype: Restricted K-Point DFT+U+V in PySCF
#
# This code extends krkspu.KRKSpU (the restricted DFT+U class) by adding
# intersite V terms. It's a simplified example and should be tested and
# adapted for production use.
#
# Author: Marco Antonio Barroca
# License: Apache 2.0 (matching PySCF)
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
# Import the restricted DFT+U parent class and helpful routines
from pyscf.pbc.dft import krkspu
from pyscf.pbc.dft.krkspu import make_minao_lo, mdot

def set_UV(mf, U_idx=None, U_val=None, V_idx=None, V_val=None):
    """
    Attach both on-site U and intersite V parameters to the mean-field object.

    Parameters
    ----------
    mf : Mean-field object (KRKSpUV below)
    U_idx : list
        Onsite orbital indices or orbital labels for the U correction.
    U_val : list
        List of floats for each U_idx element. (effective U)
    V_idx : list
        Pairs of local orbital indices for the intersite V correction, e.g.:
            [ ([3,4,5], [6,7,8]), ... ]
    V_val : list
        List of floats for each pair in V_idx. (effective V)
    """
    if U_idx is None:
        U_idx = []
    if U_val is None:
        U_val = []
    if V_idx is None:
        V_idx = []
    if V_val is None:
        V_val = []

    mf.U_idx = U_idx
    mf.U_val = U_val
    mf.U_lab = []

    mf.V_idx = V_idx
    mf.V_val = V_val
    mf.V_lab = []
    return mf


def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    r"""
    Compute the effective potential for restricted k-point DFT+U+V:
      v_eff = (v_Coulomb + v_XC) + v_U + v_V

    We rely on the parent KRKSpU for the on-site U part. Here, we add an
    intersite V correction.

    - The restricted DM has shape (nkpts, nao, nao), no separate spin dimension.
    - The local orbitals similarly are (nkpts, nao, nlo).

    Intersite E_V ~ sum_{(I,J)} [ V_{IJ} * Tr(P_I P_J) ] (simple example).
    The potential is added to vxc in AO basis.

    Parameters
    ----------
    mf : KRKSpUV object
    dm : ndarray, shape (nkpts, nao, nao)
    """
    if cell is None:
        cell = self.cell
    if dm is None:
        dm = self.make_rdm1()
    if kpts is None:
        kpts = self.kpts

    # Call the parent's get_veff explicitly
    vxc = super(KRKSpUV, self).get_veff(cell, dm, dm_last=dm_last,
                                        vhf_last=vhf_last, hermi=hermi,
                                        kpts=kpts, kpts_band=kpts_band)

    # If no V parameters, just return
    if not getattr(mf, 'V_idx', None) or not getattr(mf, 'V_val', None):
        if not hasattr(vxc, 'E_V'):
            vxc = lib.tag_array(vxc, E_V=0.0)
        return vxc

    C_ao_lo = mf.C_ao_lo  # shape (nkpts, nao, nlo) for restricted case
    if C_ao_lo.ndim != 3:
        raise ValueError("For KRKSpUV, C_ao_lo must be (nkpts, nao, nlo). "
                         f"Got {C_ao_lo.shape}.")

    ovlp = mf.get_ovlp()
    nkpts = len(kpts)

    # Build local RDM in these LOs, shape (nkpts, nlo, nlo)
    nlo = C_ao_lo.shape[-1]
    rdm1_lo = np.zeros((nkpts, nlo, nlo), dtype=np.complex128)
    for k in range(nkpts):
        C_inv = np.dot(C_ao_lo[k].conj().T, ovlp[k])
        rdm1_lo[k] = mdot(C_inv, dm[k], C_inv.conj().T)

    # Weight from IBZ if applicable
    weight = getattr(kpts, "weights_ibz", np.repeat(1.0/nkpts, nkpts))

    # Intersite energy
    E_V = 0.0

    # For each pair, apply a naive E_V ~ sum_{i in I, j in J} V * P_Ii P_Jj
    for (orbs1, orbs2), val_V in zip(mf.V_idx, mf.V_val):
        # For a.u. vs eV, do any conversion if needed:
        # val_V *= 1./27.2114  # if your input was in eV

        mesh1 = np.ix_(orbs1, orbs1)
        mesh2 = np.ix_(orbs2, orbs2)

        for k in range(nkpts):
            P1 = rdm1_lo[k][mesh1]
            P2 = rdm1_lo[k][mesh2]
            cross12 = np.einsum("ij,ji->", P1, P2)
            E_V += weight[k] * val_V * cross12

            # Potential piece: dE/dP1 = val_V * P2, etc.
            SC1 = np.dot(ovlp[k], C_ao_lo[k][:, orbs1])
            SC2 = np.dot(ovlp[k], C_ao_lo[k][:, orbs2])
            dV1 = mdot(SC1, (val_V * P2), SC1.conj().T)
            dV2 = mdot(SC2, (val_V * P1), SC2.conj().T)

            # Add to vxc (shape: (nkpts, nao, nao)) in AO basis
            vxc[k] += (dV1 + dV2).astype(vxc[k].dtype, copy=False)

    # Tag the final E_V
    old_Ev = getattr(vxc, 'E_V', 0.0)
    vxc = lib.tag_array(vxc, E_V=old_Ev + E_V.real)
    return vxc


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    r"""
    DFT+U+V total energy for the restricted k-point scenario:
      E = e1 + ecoul + exc + E_U + E_V
    """
    if dm_kpts is None:
        dm_kpts = mf.make_rdm1()
    if h1e_kpts is None:
        h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(h1e_kpts)
    weight = getattr(mf.kpts, "weights_ibz", np.repeat(1.0/nkpts, nkpts))

    # e1 = sum_k w_k Tr[h1e_kpts[k] * dm_kpts[k]]
    e1 = np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts)

    ecoul = getattr(vhf, 'ecoul', 0.0)
    exc   = getattr(vhf, 'exc',   0.0)
    E_U   = getattr(vhf, 'E_U',   0.0)
    E_V   = getattr(vhf, 'E_V',   0.0)

    etot = e1 + ecoul + exc + E_U + E_V

    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exc.real
    mf.scf_summary['E_U'] = E_U.real
    mf.scf_summary['E_V'] = E_V.real

    logger.debug(mf, 'krkspuv energy_elec: e1=%s  ecoul=%s  exc=%s  E_U=%s  E_V=%s',
                 e1, ecoul, exc, E_U, E_V)
    return etot.real, (ecoul + exc + E_U + E_V)


class KRKSpUV(krkspu.KRKSpU):
    """
    KRKSpUV: Restricted k-point DFT with on-site U + intersite V.

    Inherits from krkspu.KRKSpU and modifies get_veff + energy_elec to
    include intersite V corrections.

    Parameters
    ----------
    cell : :class:`Cell`
    kpts : ndarray
        k-point mesh
    U_idx, U_val : on-site U definition
    V_idx, V_val : intersite V definition
    C_ao_lo : local orbitals, shape (nkpts, nao, nlo) or 'minao'
    """

    # Add V-related keywords to recognized set
    _keys = krkspu.KRKSpU._keys.union({"V_idx", "V_val", "V_lab"})

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = lib.invalid_method('to_hf')

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=None, U_val=None,
                 V_idx=None, V_val=None,
                 C_ao_lo='minao', minao_ref='MINAO',
                 **kwargs):
        super(KRKSpUV, self).__init__(cell, kpts, xc=xc, exxdiv=exxdiv, **kwargs)
        set_UV(self, U_idx, U_val, V_idx, V_val)

        # Build or set local orbitals:
        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError("Only 'minao' local orbitals are implemented.")
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)

        # For restricted spin, shape should be (nkpts, nao, nlo)
        if self.C_ao_lo.ndim == 2:
            # Possibly missing the nlo dimension
            raise ValueError("C_ao_lo must have at least 3 dims (nkpts, nao, nlo).")
        elif self.C_ao_lo.ndim == 3:
            # Good, do nothing
            pass
        else:
            raise ValueError(f"Unexpected shape of C_ao_lo: {self.C_ao_lo.shape}")

    def nuc_grad_method(self):
        # Not implemented
        raise NotImplementedError


if __name__ == '__main__':
    # Quick example usage
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,0.,0.; C 0.8917,0.8917,0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 5
    cell.build()

    kmesh = [2,2,2]
    kpts = cell.make_kpts(kmesh, wrap_around=True)

    U_idx = ["1 C 2p"]
    U_val = [5.0]

    # Suppose local orbitals for site0: [3,4,5], site1: [6,7,8]
    V_idx = [([3,4,5], [6,7,8])]
    V_val = [1.0]

    mf = KRKSpUV(cell, kpts,
                 U_idx=U_idx, U_val=U_val,
                 V_idx=V_idx, V_val=V_val,
                 C_ao_lo='minao', minao_ref='gth-szv')
    mf.conv_tol = 1e-8
    e_tot = mf.kernel()
    print("Restricted DFT+U+V total energy =", e_tot)