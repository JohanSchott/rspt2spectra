#!/usr/bin/env python3

"""

d4h
===

This module contains functions which are useful for materials with D4h or Oh symmetry.

"""

import numpy as np
import sys


# Functions related to crystal-field splitting of localized orbitals


def get_CF_mean_energies(w, pdos_eg, pdos_t2g, wmin, wmax):
    """
    Return the crystal-field splitting in cubic environment.

    Symmetries can be octahedral (O_h) or tetragonal (T_d).

    """
    ega = energies.cog(w, pdos_eg, wmin, wmax)
    t2ga = energies.cog(w, pdos_t2g, wmin, wmax)
    return ega - t2ga, ega, t2ga


def get_D4h_splitting(w, pdos_x2y2, pdos_z2, pdos_xy, pdos_xz,
                      wmin, wmax):
    """
    Return the crystal-field splitting in D_4h symmetry.

    Quadradic planar systems or tetragonally distored
    octahedrons have this symmetry.

    """
    x2y2 = energies.cog(w, pdos_x2y2, wmin, wmax)
    z2 = energies.cog(w, pdos_z2, wmin, wmax)
    xy = energies.cog(w, pdos_xy, wmin, wmax)
    xz = energies.cog(w, pdos_xz, wmin, wmax)
    e_mean = (x2y2 + z2 + xy + 2 * xz) / 5.
    deltao = 1 / 2. * x2y2 + 1 / 2. * z2 - 1 / 3. * xy - 2 / 3. * xz
    delta1 = x2y2 - z2
    delta2 = xy - xz
    return e_mean, deltao, delta1, delta2


def get_delta_o(e):
    r"""
    Return CF parameter: :math:`\delta_o = e_{e_g} - e_{t_{2g}}`.

    Parameters
    ----------
    e : (N) array
        On-site energies.

    """
    if len(e) == 2:
        # Assumed order: eg and t2g
        eg = np.array(e[0])
        t2g = np.array(e[1])
        deltao = eg - t2g
    elif len(e) == 3:
        # Assumed order: eg1, eg2 and t2g
        eg1 = np.array(e[0])
        eg2 = np.array(e[1])
        t2g = np.array(e[2])
        deltao = (eg1 + eg2) / 2. - t2g
    elif len(e) == 4:
        # Assumed order: z2, x2y2, xz=yz and xy
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        xz = np.array(e[2])
        xy = np.array(e[3])
        deltao = (z2 + x2y2) / 2. - (2 * xz + xy) / 3.
    else:
        sys.exit("Wrong size of e")
    return deltao


def get_delta1(e):
    r"""
    Return CF parameter: :math:`\delta_1 = e_{x^2-y^2} - e_{z^2}`.

    Works in O_h and D_4h symmetry.

    Parameters
    ----------
    e : (N) array
        On-site energies.

    """
    if len(e) == 2:
        # Assumed order: eg and t2g
        eg = np.array(e[0])
        # t2g = np.array(e[1])
        delta1 = np.zeros_like(eg)
    elif len(e) == 3:
        # Assumed order: z2, x2y2 and t2g
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        delta1 = x2y2 - z2
    elif len(e) == 4:
        # Assumed order: z2, x2y2 and xz=yz and xy
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        delta1 = x2y2 - z2
    else:
        sys.exit("Wrong size of e")
    return delta1


def get_delta2(e):
    r"""
    Return CF parameter: :math:`\delta_2 = e_{xy} - e_{xz}`.

    Works in O_h and D_4h symmetry.

    Parameters
    ----------
    e : (N) array
        On-site energies.

    """
    if len(e) == 2:
        # assumed order: eg and t2g
        eg = np.array(e[0])
        # t2g = np.array(e[1])
        delta2 = np.zeros_like(eg)
    elif len(e) == 3:
        # assumed order: z2, x2y2 and t2g
        z2 = np.array(e[0])
        delta2 = np.zeros_like(z2)
    elif len(e) == 4:
        # assumed order: z2, x2y2 and xz=yz and xy
        # z2 = np.array(e[0])
        # x2y2 = np.array(e[1])
        xz = np.array(e[2])
        xy = np.array(e[3])
        delta2 = xy - xz
    else:
        sys.exit("Wrong size of e")
    return delta2


def get_3d_energy(e):
    r"""
    Return CF parameter: average 3d on-site energy.

    Works in O_h and D_4h symmetry.

    Parameters
    ----------
    e : (N) array
        On-site energies.

    """
    if len(e) == 2:
        # Assumed order: eg and t2g
        eg = np.array(e[0])
        t2g = np.array(e[1])
        em = 1 / 5. * (2 * eg + 3 * t2g)
    elif len(e) == 3:
        # Assumed order: eg1, eg2 and t2g
        eg1 = np.array(e[0])
        eg2 = np.array(e[1])
        t2g = np.array(e[2])
        em = 1 / 5. * (1 * eg1 + 1 * eg2 + 3 * t2g)
    elif len(e) == 4:
        # Assumed order: z2, x2y2, xz=yz and xy
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        xz = np.array(e[2])
        xy = np.array(e[3])
        em = 1 / 5. * (1 * z2 + 1 * x2y2 + 2 * xz + 1 * xy)
    else:
        sys.exit("Wrong size of e")
    return em


# Crystal-field operators and functions, expressed in 3d spherical harmonics basis.
e = {}
# One formulation of the crystal-field Hamiltonian matrices
# Dq
e['q'] = np.zeros((5, 5))
np.fill_diagonal(e['q'], [1, -4, 6, -4, 1])
e['q'][-1, 0] = 5
e['q'][0, -1] = 5
# Ds
e['s'] = np.zeros((5, 5))
np.fill_diagonal(e['s'], [2, -1, -2, -1, 2])
# Dt
e['t'] = np.zeros((5, 5))
np.fill_diagonal(e['t'], [-1, 4, -6, 4, -1])

# Another formulation of the crystal-field Hamiltonian matrices
e['o'] = 1/10.*e['q']
e['1'] = 1/20.*e['q'] + 1/7.*e['s'] + 3/35.*e['t']
e['2'] = -1/15.*e['q'] + 1/7.*e['s'] - 4/35.*e['t']

# Another formulation of the crystal-field Hamiltonian matrices
e['eg'] = np.zeros((5, 5))
np.fill_diagonal(e['eg'], [0.5, 0, 1, 0, 0.5])
e['eg'][0, -1] = 0.5
e['eg'][-1, 0] = 0.5
e['t2g'] = np.zeros((5, 5))
np.fill_diagonal(e['t2g'], [0.5, 1, 0, 1, 0.5])
e['t2g'][0, -1] = -0.5
e['t2g'][-1, 0] = -0.5

# Another formulation of the crystal-field Hamiltonian matrices
e['z2'] = np.zeros((5, 5))
np.fill_diagonal(e['z2'], [0, 0, 1, 0, 0])
e['x2y2'] = np.zeros((5, 5))
np.fill_diagonal(e['x2y2'], [0.5, 0, 0, 0, 0.5])
e['x2y2'][0, -1] = 0.5
e['x2y2'][-1, 0] = 0.5
e['xz_yz'] = np.zeros((5, 5))
np.fill_diagonal(e['xz_yz'], [0, 1, 0, 1, 0])
e['xy'] = np.zeros((5, 5))
np.fill_diagonal(e['xy'], [0.5, 0, 0, 0, 0.5])
e['xy'][0, -1] = -0.5
e['xy'][-1, 0] = -0.5


def get_Dq_Ds_Dt(deltao, delta1, delta2):
    """
    Convert crystal-field parameters from one representation to another.

    Parameters
    ----------
    deltao : float
        Energy separation between e_g and t_2g orbitals.
    delta1 : float
        Energy separation between x^2-y^2 and z^2 orbitals.
    delta2 : float
        Energy separation between xy and xz orbitals (xz same as yz).

    Returns
    -------
    dq : float
    ds : float
    dt : float

    """
    dq = 1 / 10. * deltao + 1 / 60. * (3 * delta1 - 4 * delta2)
    ds = (delta1 + delta2) / 7.
    dt = (3 * delta1 - 4 * delta2) / 35.
    return dq, ds, dt


def get_deltao_delta1_delta2(dq, ds, dt):
    """
    Convert crystal-field parameters from one representation to another.

    Parameters
    ----------
    dq : float
    ds : float
    dt : float

    Returns
    -------
    deltao : float
        Energy separation between e_g and t_2g orbitals.
    delta1 : float
        Energy separation between x^2-y^2 and z^2 orbitals.
    delta2 : float
        Energy separation between xy and xz orbitals (xz same as yz).

    """
    deltao = 10 * dq
    delta1 = 4 * ds + 5 * dt
    delta2 = 3 * ds - 5 * dt
    return deltao, delta1, delta2


def get_CF_hamiltonian_with_Dopp(e_mean, dq, ds, dt):
    """
    Returns crystal-field Hamiltonian in spherical harmonics basis.
    """
    h = e_mean*np.eye(5) + dq * e['q'] + ds * e['s'] + dt * e['t']
    return h


def get_CF_hamiltonian_with_delta(e_mean, deltao, delta1, delta2):
    """
    Returns crystal-field hamiltonian in spherical harmonics basis.
    """
    h = (e_mean*np.eye(5) + deltao * e['o'] +
         delta1 * e['1'] + delta2 * e['2'])
    return h


def get_CF_hamiltonian_eg_t2g(e_eg, e_t2g):
    """
    Returns crystal-field hamiltonian in spherical harmonics basis.
    """
    h_eg = e_eg * e['eg']
    h_t2g = e_t2g * e['t2g']
    h = h_eg + h_t2g
    return h


def get_CF_hamiltonian_cubic(e_z2, e_x2y2, e_xz_yz, e_xy):
    """
    Returns crystal-field hamiltonian in spherical harmonics basis.
    """
    h_z2 = e_z2 * e['z2']
    h_x2y2 = e_x2y2 * e['x2y2']
    h_xz_yz = e_xz_yz * e['xz_yz']
    h_xy = e_xy * e['xy']
    # Combine hybridization terms
    h = h_z2 + h_x2y2 + h_xz_yz + h_xy
    return h
