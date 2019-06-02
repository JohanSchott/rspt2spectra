#!/usr/bin/env python3

"""

dc
==

This module contains double counting functions.

"""


def dc_MLFT(n3d_i, c, Fdd, n2p_i=None, Fpd=None, Gpd=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.

    Parameters
    ----------
    n3d_i : int
        Nominal (integer) 3d occupation.
    c : float
        Many-body correction to the charge transfer energy.
    n2p_i : int
        Nominal (integer) 2p occupation.
    Fdd : list
        Slater integrals {F_{dd}^k}, k \in [0,1,2,3,4]
    Fpd : list
        Slater integrals {F_{pd}^k}, k \in [0,1,2]
    Gpd : list
        Slater integrals {G_{pd}^k}, k \in [0,1,2,3]

    Notes
    -----
    The `c` parameter is related to the charge-transfer
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if not int(n3d_i) == n3d_i:
        raise ValueError('3d occupation should be an integer')
    if n2p_i is not None and int(n2p_i) != n2p_i:
        raise ValueError('2p occupation should be an integer')
    # Average repulsion energy defines Udd and Upd
    Udd = Fdd[0] - 14/441*(Fdd[2] + Fdd[4])
    if n2p_i is None and Fpd is None and Gpd is None:
        return Udd * n3d_i - c
    if n2p_i == 6 and Fpd is not None and Gpd is not None:
        Upd = Fpd[0] - (1/15)*Gpd[1] - (3/70)*Gpd[3]
        return [Udd*n3d_i + Upd*n2p_i - c, Upd*(n3d_i + 1) - c]
    else:
        raise ValueError('double counting input wrong.')


def dc_FLL(n, F0, F2, F4):
    r"""
    Return double counting in the fully localized limit.

    Parameters
    ----------
    n : float
        Occupation of 3d orbitals.
    F0 : float
        Slater integral :math:`F_{dd}^{(0)}`.
    F2 : float
        Slater integral :math:`F_{dd}^{(2)}`.
    F4 : float
        Slater integral :math:`F_{dd}^{(4)}`.

    """
    J = 1/14*(F2 + F4)
    return F0 * (n - 1/2) - J/2*(n - 1)
