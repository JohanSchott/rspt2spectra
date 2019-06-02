#!/usr/bin/env python3

r"""

Unitary Tranformation
=====================

This module contains functions which are useful for
unitary transformations.

Unitary transformation of spherical to cubic harmonics.
Transform observables, e.g. the crystal field operator, from the spherical harmonics to cubic harmonics basis.

d-orbitals.

The d-cubic harmonics expressed in 3-d spherical harmonics:

.. math:: 0: d_{z^2} = Y_2^{0}

.. math:: 1: d_{x^2-y^2} = \frac{1}{\sqrt{2}}Y_2^{-2} + \frac{1}{\sqrt{2}}Y_2^{2}

.. math:: 2: d_{yz} = \frac{-i}{\sqrt{2}}Y_2^{-1} + \frac{-i}{\sqrt{2}}Y_2^{1}

.. math:: 3: d_{xz} = \frac{1}{\sqrt{2}}Y_2^{-1} + \frac{-1}{\sqrt{2}}Y_2^{1}

.. math:: 4: d_{xy} = \frac{i}{\sqrt{2}}Y_2^{-2} + \frac{-i}{\sqrt{2}}Y_2^{2}

This is the ordering used in RSPt.

Let's order also the spherical harmonic orbitals (in the order it's done in RSPt):

.. math:: 0: Y_{2}^{-2}

.. math:: 1: Y_{2}^{-1}

.. math:: 2: Y_{2}^{0}

.. math:: 3: Y_{2}^{1}

.. math:: 4: Y_{2}^{2}

Now we define for convienience the notation: :math:`d_i` and :math:`Y_{d,i}` for the :math:`i` cubic and
spherical harmonic d-orbital respectively.

This makes it easy to write down a transformation matrix :math:`u_d`, with element :math:`u_{d,(i,j)}` representing
the contribution of spherical harmonic :math:`i` to the cubic harmonic :math:`j`:

.. math:: \lvert d_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)} \lvert Y_{d,i} \rangle, \; \mathrm{for} \; j \in \{0,1,2,3,4\},

with

.. math::  u_d =  \begin{bmatrix}  0  & \frac{1}{\sqrt{2}} & 0 & 0 & \frac{i}{\sqrt{2}}  \\ 0  & 0 & \frac{-i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\    1  & 0 & 0 & 0 & 0 \\     0  & 0 & \frac{-i}{\sqrt{2}} & \frac{-1}{\sqrt{2}} & 0 \\    0       & \frac{1}{\sqrt{2}}   & 0 & 0 & \frac{-i}{\sqrt{2}}  \end{bmatrix}

Note: :math:`u_d` is a unitary matrix, hence:

.. math:: u_d^\dagger u_d = I

.. math:: u_d u_d^\dagger = I

Orthogonality of spherical harmonics:

.. math:: \delta_{i,j} = \langle Y_{d,i} \lvert Y_{d,j} \rangle

Orthogonality of cubic harmonics:

.. math:: \delta_{i,j} = \langle d_i \lvert d_j \rangle  = \sum_{k=0}^4 \bar{u}_{d,(k,i)} \langle Y_{d,k} \lvert \sum_{n=0}^4 u_{d,(n,j)} \lvert Y_{d,n} \rangle \\ = \sum_{k,n} \bar{u}_{d,(k,i)} u_{d,(n,j)} \delta_{k,n} = \sum_{k=0}^4 \bar{u}_{d,(k,i)} u_{d,(k,j)} \\ = \sum_{k=0}^4 (u_d^\dagger)_{(i,k)} u_{d,(k,j)},

which in matrix form reads:

.. math:: I = u_d^\dagger u_d.

An observable, expressed in cubic harmonics basis is:

.. math:: \tilde{A}_{i,j} = \langle d_i \lvert \hat{A} \rvert d_j \rangle = \sum_{k=0}^4 \bar{u}_{d,(k,i)} \langle Y_{d,k} \lvert \hat{A} \sum_{n=0}^4 u_{d,(n,j)} \lvert Y_{d,n} \rangle  \\ = \sum_{k,n} \bar{u}_{d,(k,i)} u_{d,(n,j)} \langle Y_{d,k} \lvert \hat{A} \rvert Y_{d,n} \rangle  = \sum_{k,n} \bar{u}_{d,(k,i)} A_{k,n} u_{d,(n,j)} \\ = \sum_{k=0}^4 (u_d^\dagger)_{(i,k)} \sum_{n=0}^4 A_{k,n} u_{d,(n,j)},

which in matrix form becomes:

.. math:: \tilde{A} = u_d^\dagger A u_d.

Thus, transforming an observable from spherical to cubic harmonic basis is
simple done by multiplying its matrix representation in the spherical
harmonics basis from left :math:`u_d^\dagger` and from the right
with :math:`u_d`.

Due to the orthogonality of the columns and the rows in :math:`u_d`, we can also easily transform back:

.. math:: A = u_d \tilde{A} u_d^\dagger.

p-orbitals.

The p-cubic harmonics expressed in 3-d spherical harmonics:

.. math:: 0: p_y = \frac{i}{\sqrt{2}}Y_1^{-1} + \frac{i}{\sqrt{2}}Y_1^{1}

.. math:: 1: p_x = \frac{1}{\sqrt{2}}Y_1^{-1} - \frac{1}{\sqrt{2}}Y_1^{1}

.. math:: 2: p_z = Y_1^{0}

The ordering is the same as in RSPt.

Let's order also the spherical harmonic orbitals (in the order it's done in RSPt):

.. math:: 0: Y_1^{-1}

.. math:: 1: Y_1^{0}

.. math:: 2: Y_1^{1}

Now we define for convienience the notation: :math:`p_i` and :math:`Y_{p,i}`
for the :math:`i` cubic and spherical harmonic respectively.

This makes it easy to write down a transformation matrix :math:`u_p`,
with element :math:`u_{p,(i,j)}` representing the contribution of
spherical harmonics :math:`i` to the cubic harmonic :math:`j`:

.. math:: \lvert d_j \rangle  = \sum_{i=0}^4 u_{p,(i,j)} \lvert Y_{p,i} \rangle, \; \mathrm{for} \; j \in \{0,1,2,3,4\},

with

.. math:: u_p =  \begin{bmatrix}    \frac{i}{\sqrt{2}}       & \frac{1}{\sqrt{2}} & 0  \\    0       & 0 & 1  \\     \frac{i}{\sqrt{2}}       & \frac{-1}{\sqrt{2}} & 0   \end{bmatrix}

We can define a transformation matrix between cubic and spherical harmonics,
as we did for the d-orbitals.
Thus a observable in cubic harmonics becomes:
:math:`\tilde{A} = u_p^\dagger A u_p`, where :math:`u_p`
is the transformation matrix for p-orbitals.

p- and d-orbitals.

An observable with a d-oribtal as bra and a p-orbital as a ket can be
expressed in cubic harmonics:

.. math:: \tilde{A}_{i,j} = \langle d_i \lvert \hat{A} \rvert p_j \rangle = \sum_{k=0}^4 \bar{u}_{d,(k,i)} \langle Y_{d,k} \lvert \hat{A} \sum_{n=0}^2 u_{p,(n,j)} \lvert Y_{p,n} \rangle  \\ = \sum_{k,n} \bar{u}_{d,(k,i)} u_{p,(n,j)} \langle Y_{d,k} \lvert \hat{A} \rvert Y_{p,n} \rangle  = \sum_{k,n} \bar{u}_{d,(k,i)} A_{k,n} u_{p,(n,j)} \\ = \sum_{k=0}^4 (u_d^\dagger)_{(i,k)} \sum_{n=0}^2 A_{k,n} u_{p,(n,j)},

which in matrix form becomes:

.. math:: \tilde{A} = u_d^\dagger A u_p.

Thus, transforming an observable from spherical to cubic harmonic basis
is simple done by multiplying its matrix representation in the
spherical harmonics basis from left
:math:`u_d^\dagger` and from the right with :math:`u_p`.

Due to the orthogonality of the columns and the rows in
:math:`u_d` and :math:`u_p`, we can also easily transform back:

.. math:: A = u_d \tilde{A} u_p^\dagger.


"""

import numpy as np
import sys


def rotate(d, r_left, r_right):
    r"""
    Return matrix rotated from left and right.

    Assumptions:
    - Variable `d` to contain both spin channels
    but with no spin off-diagonal elements.
    - Variables `r_left` and `r_right` contain only one spin channel.
    The other spin channel has to be the same.

    Parameters
    ----------
    d : (N,M) array
        Spin-orbitals are ordered: 0dn, 0up, 1dn, 1up, ...
    r_left : (N/2,N/2) array
    r_right : (M/2,M/2) array

    Returns
    -------
    dp : (N,M) ndarray
        Spin-orbitals are ordered: 0dn, 0up, 1dn, 1up, ...

    """
    # Remove spin
    dm = d[::2, ::2]
    # Rotate to spherical harmonics basis
    # in the rotated coordinate system
    dmp = np.dot(np.transpose(np.conj(r_left)), np.dot(dm, r_right))
    # Add spin
    dp = np.zeros(2*np.array(np.shape(dmp)), dtype=np.complex)
    dp[::2, ::2] = dmp
    dp[1::2, 1::2] = dmp
    return dp


def get_spherical_2_cubic_matrix(spinpol=False, l=2):
    r"""
    Return unitary ndarray for transforming from spherical
    to cubic harmonics.

    Parameters
    ----------
    spinpol : boolean
        If transformation involves spin.
    l : integer
        Angular momentum number. p: l=1, d: l=2.

    Returns
    -------
    u : (M,M) ndarray
        The unitary matrix from spherical to cubic harmonics.

    Notes
    -----
    Element :math:`u_{i,j}` represents the contribution of spherical
    harmonics :math:`i` to the cubic harmonic :math:`j`:

    .. math:: \lvert l_j \rangle  = \sum_{i=0}^4 u_{d,(i,j)} \lvert Y_{d,i} \rangle.

    """
    if l == 1:
        u = np.zeros((3, 3), dtype=np.complex)
        u[0, 0] = 1j/np.sqrt(2)
        u[2, 0] = 1j/np.sqrt(2)
        u[0, 1] = 1/np.sqrt(2)
        u[2, 1] = -1/np.sqrt(2)
        u[1, 2] = 1
    elif l == 2:
        u = np.zeros((5, 5), dtype=np.complex)
        u[2, 0] = 1
        u[[0, -1], 1] = 1/np.sqrt(2)
        u[1, 2] = -1j/np.sqrt(2)
        u[-2, 2] = -1j/np.sqrt(2)
        u[1, 3] = 1/np.sqrt(2)
        u[-2, 3] = -1/np.sqrt(2)
        u[0, 4] = 1j/np.sqrt(2)
        u[-1, 4] = -1j/np.sqrt(2)
    else:
        sys.exit('This angular momentum is not implemented yet')
    if spinpol:
        n, m = np.shape(u)
        uSpin = np.zeros((2*n, 2*m), dtype=np.complex)
        uSpin[:n, :m] = u
        uSpin[n:, m:] = u
        u = uSpin
    return u


def get_spinpol_and_ls(n, m):
    r"""
    Calculate if spinpolarized and which angular momentum
    the matrix dimensions n,m corresponds to.

    Parameters
    ----------
    n : integer
        Number of elements in the first dimension.
    m : integer
        Number of elements in the second dimension.

    Returns
    -------
    spinpol : boolean
              if matrix is spinpolarized
    ls : list of two integers
         angular momentum numbers

    """
    if n % 2 == 0:
        spinpol = True
        ls = [(n/2-1)/2, (m/2-1)/2]
    else:
        spinpol = False
        ls = [(n-1)/2, (m-1)/2]
    return spinpol, ls


def spherical_2_cubic(a):
    r"""
    Create the observable matrix expressed in cubic harmonics
    instead of spherical harmonics.

    .. math:: \tilde{a} = u^\dagger a u.

    Parameters
    ----------
    a : matrix
        Observable, expressed in spherical harmonics.

    Returns
    -------
    at : matrix
         Observable, expressed in cubic harmonics.

    """
    spinpol, ls = get_spinpol_and_ls(*np.shape(a))
    u_bra = get_spherical_2_cubic_matrix(spinpol=spinpol, l=ls[0])
    u_ket = get_spherical_2_cubic_matrix(spinpol=spinpol, l=ls[1])
    at = np.dot(np.transpose(np.conj(u_bra)),np.dot(a, u_ket))
    return at


def cubic_2_spherical(at):
    r"""
    Create the observable matrix expressed in spherical
    harmonics instead of cubic harmonics.

    .. math:: a = u \tilde{a} u^\dagger.

    Parameters
    ----------
    at : matrix
         Observable, expressed in cubic harmonics.

    Returns
    -------
    a : matrix
        Observable, expressed in spherical harmonics.

    """
    spinpol, ls = get_spinpol_and_ls(*np.shape(at))
    u_bra = get_spherical_2_cubic_matrix(spinpol=spinpol, l=ls[0])
    u_ket = get_spherical_2_cubic_matrix(spinpol=spinpol, l=ls[1])
    a = np.dot(u_bra, np.dot(at,np.transpose(np.conj(u_ket))))
    return a
