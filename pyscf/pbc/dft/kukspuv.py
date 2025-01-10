#!/usr/bin/env python
#
# Improved Prototype: DFT+U+V in PySCF (PBC, k-point sampling).
#
# This code extends the KUKSpU functionality by adding an optional
# intersite V correction. It is provided as a PROTOTYPE and may require
# additional validation, especially regarding double-counting terms,
# phase factors for k-point symmetries, and the choice of local orbitals.
#
# Author: Marco Antonio Barroca
# License: Apache 2.0 (following PySCF license)
#

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.pbc.dft import kuks
from pyscf.pbc.dft.krkspu import make_minao_lo, mdot

def set_UV(mf, U_idx=None, U_val=None, V_idx=None, V_val=None):
    """
    Set both onsite U and intersite V parameters to the mean-field object.

    Parameters
    ----------
    mf : Mean-field object
        Typically an instance of KUKSpUV (defined below).

    U_idx : list
        Same usage as in PySCF's DFT+U (onsite orbitals). Can be:
        - list of lists of LO indices
        - list of strings (e.g. "Ni 3d", "1 C 2p", etc.)
        - or a mix

    U_val : list of floats
        Each element is the effective U (in eV or a.u.) for the corresponding
        entry in U_idx. Must match 1-to-1 in length.

    V_idx : list
        Defines pairs of local orbitals on *different* sites for intersite V.
        For example: [ ([3,4,5], [6,7,8]), ([10,11],[12,13]) ]
        or a more general structure if you prefer strings.

    V_val : list of floats
        Each element is the intersite V (in eV or a.u.) corresponding to one
        pair in V_idx. Must match in length.

    Returns
    -------
    mf : Mean-field object
        Updated with the new attributes.
    """
    if U_idx is None: U_idx = []
    if U_val is None: U_val = []
    if V_idx is None: V_idx = []
    if V_val is None: V_val = []

    # Onsite
    mf.U_idx = U_idx
    mf.U_val = U_val
    mf.U_lab = []

    # Intersite
    mf.V_idx = V_idx
    mf.V_val = V_val
    mf.V_lab = []

    return mf


def get_veff(mf, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Compute the effective potential for DFT+U+V:
      v_eff = (v_Coulomb + v_XC) + v_U + v_V

    The v_U part is handled by the parent class (in krkspu.py).
    Here, we add an additional v_V for intersite Hubbard interactions.

    For each pair of local orbitals \((\phi_{i}, \phi_{j})\) associated with
    different sites, we apply a term:
      E_V ~ \sum_{i \in I, j \in J} V_{IJ} \, P_{i}^{(\alpha/\beta)} P_{j}^{(\alpha/\beta)} + ...
    with a derivative that contributes to the potential.

    This is highly schematic and may need to be adapted for your specific
    double-counting correction or functional form.

    Parameters
    ----------
    mf : KUKSpUV object
        The mean-field object.

    cell : :class:`Cell` object
        The simulation cell.

    dm : ndarray or list of ndarrays
        One-particle density matrix, shape = (2, nkpts, nao, nao) typically.

    Returns
    -------
    vxc : ndarray or list of ndarrays
        The total effective potential in AO basis, shape same as `dm`.
        It has attributes vxc.ecoul, vxc.exc, vxc.E_U, and now vxc.E_V.
    """
    if cell is None:
        cell = mf.cell
    if dm is None:
        dm = mf.make_rdm1()
    if kpts is None:
        kpts = mf.kpts

    # First, call the parent's get_veff, which includes normal DFT + U
    # This may come from krkspu.KUKSpU or from a super call chain.
    vxc = super(mf.__class__, mf).get_veff(
        cell, dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi,
        kpts=kpts, kpts_band=kpts_band
    )

    # If no V_idx or V_val are defined, just return
    if not getattr(mf, 'V_idx', None) or not getattr(mf, 'V_val', None):
        # Tag E_V=0 so that energy_elec can read it
        if not hasattr(vxc, 'E_V'):
            vxc = lib.tag_array(vxc, E_V=0.0)
        return vxc

    # For the intersite portion, we need the local orbital density matrices:
    C_ao_lo = mf.C_ao_lo   # shape = (2, nkpts, nao, nlo)
    if C_ao_lo.ndim != 4:
        raise ValueError("C_ao_lo must have shape (2, nkpts, nao, nlo). Got %s" 
                         % (C_ao_lo.shape,))

    ovlp = mf.get_ovlp()
    nkpts = len(kpts)
    nspin = 2  # For UHF-based approach

    # Build local RDM in these LOs
    # rdm1_lo[s, k] -> shape = (nlo, nlo) for spin s, k-point k
    rdm1_lo = np.zeros((nspin, nkpts, C_ao_lo.shape[-1], C_ao_lo.shape[-1]),
                       dtype=np.complex128)
    for s in range(nspin):
        for k in range(nkpts):
            C_inv = np.dot(C_ao_lo[s, k].conj().T, ovlp[k])
            rdm1_lo[s, k] = mdot(C_inv, dm[s][k], C_inv.conj().T)

    # IBZ weighting
    is_ibz = hasattr(kpts, "kpts_ibz")
    weight = getattr(kpts, "weights_ibz", np.repeat(1.0/nkpts, nkpts))

    # Initialize E_V
    E_V = 0.0

    # Now apply each pair's V
    for (orbs1, orbs2), val_V in zip(mf.V_idx, mf.V_val):
        # Convert from eV to a.u. if needed
        # (We assume user has done so or does it themselves, or we do eV->Hartree factor)
        # val_V *= (1.0/27.2114)  # Example if user gave eV, comment out if already in a.u.

        # Construct the sub-block slices:
        # orbs1, orbs2 might be a list of indices for local orbitals
        # e.g. [3,4,5], [6,7,8]
        mesh1 = np.ix_(orbs1, orbs1)
        mesh2 = np.ix_(orbs2, orbs2)

        for s in range(nspin):
            for k in range(nkpts):
                # Sub-blocks
                P1 = rdm1_lo[s, k][mesh1]
                P2 = rdm1_lo[s, k][mesh2]

                # A simple example E_V ~ +V * Tr(P1 * P2), ignoring double counting
                # Real functionals often require more terms (e.g. -Tr(P1^2 P2^2), etc.).
                cross12 = np.einsum("ij,ji->", P1, P2)
                E_V += weight[k] * val_V * cross12

                # Potential contribution (naive derivative):
                # δ(E_V) / δP1 = V * P2 ; δ(E_V) / δP2 = V * P1
                # Projected back to AO basis
                SC1 = np.dot(ovlp[k], C_ao_lo[s, k][:, orbs1])
                SC2 = np.dot(ovlp[k], C_ao_lo[s, k][:, orbs2])
                dV1 = mdot(SC1, (val_V * P2), SC1.conj().T)
                dV2 = mdot(SC2, (val_V * P1), SC2.conj().T)
                vxc[s][k] += (dV1 + dV2).astype(vxc[s][k].dtype, copy=False)

    # Here, you might add a double-counting correction for E_V if your functional
    # calls for it. For a more advanced approach, define a separate function:
    #
    #   E_V_dc = something(...) 
    #   E_V -= E_V_dc
    #   vxc[s][k] -= derivative_of(E_V_dc)
    #
    # which you handle similarly as above. 

    # Attach E_V to vxc for later retrieval in energy_elec
    # If the parent code already put E_V, we add to it
    old_E_V = getattr(vxc, 'E_V', 0.0)
    E_V_total = old_E_V + E_V.real
    vxc = lib.tag_array(vxc, E_V=E_V_total)

    return vxc


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    r"""
    Compute the total electronic energy for DFT+U+V.

    E_elec = (e1 + ecoul + exc) + E_U + E_V

    Where:
    - e1    = \(\sum_{k,s} w_k Tr[h_{k} P_{k}^{s}]\)
    - ecoul = classical Coulomb (J)
    - exc   = XC functional energy
    - E_U   = onsite Hubbard correction from DFT+U
    - E_V   = intersite Hubbard correction from DFT+U+V

    Parameters
    ----------
    mf : KUKSpUV object
        The mean-field object.
    dm_kpts : ndarray
        Density matrix (optional, by default from mf.make_rdm1()).
    h1e_kpts : ndarray
        1-electron Hamiltonian in k-space (optional).
    vhf : ndarray
        Already-computed effective potential (optional).

    Returns
    -------
    tot_e : float
        The real part of the total electronic energy.
    ecoul_exc_eu_ev : float
        The sum (ecoul + exc + E_U + E_V) as a convenience.
    """
    if h1e_kpts is None:
        h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None:
        dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)

    weight = getattr(mf.kpts, "weights_ibz",
                     np.array([1.0/len(h1e_kpts),]*len(h1e_kpts)))

    # 1-electron term
    e1 = (np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[0]) +
          np.einsum('k,kij,kji', weight, h1e_kpts, dm_kpts[1]))

    # Retrieve the pieces from vhf
    ecoul = getattr(vhf, 'ecoul', 0.0)
    exc   = getattr(vhf, 'exc',   0.0)
    E_U   = getattr(vhf, 'E_U',   0.0)
    E_V   = getattr(vhf, 'E_V',   0.0)

    tot_e = e1 + ecoul + exc + E_U + E_V

    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = ecoul.real
    mf.scf_summary['exc'] = exc.real
    mf.scf_summary['E_U'] = E_U.real
    mf.scf_summary['E_V'] = E_V.real

    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s  E_U = %s  E_V = %s',
                 e1, ecoul, exc, E_U, E_V)

    return tot_e.real, (ecoul + exc + E_U + E_V)


class KUKSpUV(kuks.KUKS):
    """
    KUKSpUV: DFT+U+V class with k-point sampling for PBC systems.

    This extends PySCF's KUKS with onsite U and intersite V corrections.

    Parameters
    ----------
    cell : :class:`Cell`
        Simulation cell object from PySCF.
    kpts : ndarray
        List/array of k-points in fractional or Cartesian coords.
    xc : str
        Exchange-correlation functional label, e.g. 'LDA,VWN'
    exxdiv : str
        Exchange divergence treatment for PBC. Default: 'ewald'
    U_idx, U_val : see set_UV
    V_idx, V_val : see set_UV
    C_ao_lo : str or ndarray
        Local orbitals to be used for the U/V projection. 
        - If 'minao', uses a minimal basis approach from PySCF.
        - If ndarray, shape must be (2, nkpts, nao, nlo) or broadcastable to it.
    minao_ref : str
        Reference basis for the minao construction. Default: 'MINAO'.
    """

    _keys = {"U_idx", "U_val", "C_ao_lo", "U_lab", 
             "V_idx", "V_val", "V_lab"}

    get_veff = get_veff
    energy_elec = energy_elec
    to_hf = lib.invalid_method('to_hf')  # Not implemented

    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald'),
                 U_idx=None, U_val=None,
                 V_idx=None, V_val=None,
                 C_ao_lo='minao', minao_ref='MINAO',
                 **kwargs):
        super().__init__(cell, kpts, xc=xc, exxdiv=exxdiv, **kwargs)

        set_UV(self, U_idx, U_val, V_idx, V_val)

        # Build or read local orbitals for the U+V projection
        if isinstance(C_ao_lo, str):
            if C_ao_lo.upper() == 'MINAO':
                self.C_ao_lo = make_minao_lo(self, minao_ref)
            else:
                raise NotImplementedError(f"Only 'MINAO' LO generation is implemented. Got {C_ao_lo}")
        else:
            self.C_ao_lo = np.asarray(C_ao_lo)

        # Ensure shape is (2, nkpts, nao, nlo)
        if self.C_ao_lo.ndim == 3:
            # If shape is (nkpts, nao, nlo), broadcast to spin=2
            self.C_ao_lo = np.asarray((self.C_ao_lo, self.C_ao_lo))
        elif self.C_ao_lo.ndim == 4:
            # Possibly (1, nkpts, nao, nlo) or (2, ...)
            if self.C_ao_lo.shape[0] == 1:
                self.C_ao_lo = np.asarray((self.C_ao_lo[0], self.C_ao_lo[0]))
            if self.C_ao_lo.shape[0] != 2:
                raise ValueError("C_ao_lo shape mismatch. Must have spin dimension = 2.")
        else:
            raise ValueError("Unexpected dimensionality of C_ao_lo. "
                             f"Got shape: {self.C_ao_lo.shape}.")

    def nuc_grad_method(self):
        # Not implemented
        raise NotImplementedError


if __name__ == '__main__':
    # Example usage / test
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.verbose = 5
    cell.build()

    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh, wrap_around=True)

    # Example: Onsite U on "1 C 2p", plus an intersite V between
    # the p-orbitals of atom0 and atom1 (toy example).
    U_idx = ["1 C 2p"]
    U_val = [5.0]  # eV or a.u., be consistent

    # Suppose we know local orbitals for atoms are [3,4,5], [6,7,8]
    V_idx = [([3,4,5], [6,7,8])]
    V_val = [1.0]

    mf = KUKSpUV(cell, kpts,
                 U_idx=U_idx, U_val=U_val,
                 V_idx=V_idx, V_val=V_val,
                 minao_ref='gth-szv')
    mf.conv_tol = 1e-10

    print("Running DFT+U+V SCF with the improved prototype...")
    e_tot = mf.kernel()
    print("Final E_tot (DFT+U+V):", e_tot)