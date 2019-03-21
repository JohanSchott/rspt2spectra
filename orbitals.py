#!/usr/bin/env python3

"""

orbitals
========

This module contains functions which are relevant for the choice
of localized orbitals.

"""

import numpy as np
import sys

from rspt2spectra import unitarytransform
from rspt2spectra.energies import print_matrix

# ----------------------------------------------------------
# Helpful function when generating a rotated basis
# for the localized orbitals in RSPt

def write_proj_file(x, spinpol=False, filename='proj-LABEL-IrrXXXX.inp',
                    tol=1e-11):
    """
    Return the rotation/projection file needed to rotate
    a local basis in RSPt.

    Parameters
    ----------
    x : (N,N) array
        Eigenvectors (columnvectors) to the observable to
        diagonalize, e.g. eigenvectors to local Hamiltonian H.
    spinpol : boolean
        If spin-polarized calculations.
    filename : str
        The file name should have the format:
        'proj-LABEL-IrrXXXX.inp'
    tol : float
        Ignore values smaller than this to avoid tiny
        rotations due to noise.

    """
    n = len(x[:, 0])
    m = len(x[0, :])
    if n != m:
        sys.exit('Square matrix expected')
    counter = 0
    for j in range(m):
        for i in range(n):
            if np.abs(x[i, j]) > tol:
                counter += 1
    with open(filename, 'w') as f:
        if spinpol:
            f.write('{:d} {:d} {:d}'.format(n, n, counter))
        else:
            f.write('{:d} {:d} {:d}'.format(n, 2 * n, counter))
        f.write(' ! presize, dmftsize, #lines, each row has'
                ' (i,j,Re v[i,j],Im v[i,j]).')
        f.write('Eigenvectors on columns. \n')
        for j in range(m):
            for i in range(n):
                if np.abs(x[i, j]) > tol:
                    f.write(
                        '{:d} {:d} {:20.15f} {:20.15f} \n'.format(
                            i + 1, j + 1, x[i, j].real, x[i, j].imag))

def get_u_transformation(n, basis_tag, irr_flag, spherical_bath_basis,
                         spinpol, norb, verbose_text=False):
    """
    Return the unitary matrix to rotate to spherical harmonics basis.

    Parameters
    ----------
    n : int
        The dimensions of the Hamiltonian are (n,n).
    basis_tag : str
        Correlated orbitals to study. Examples are:
        '0102010100',
        '0102010103',
        '0102010103-obs',
        '0102010100-obs1',
        '0102010100-obs'
    irr_flag : str
        Basis keyword for projection.
    spherical_bath_basis : boolean
        If bath orbitals should be spherical harmonics or kept in rotated basis.
    spinpol : boolean
        If the system is spin-polarized.
    norb : int
        Number of impurity orbitals.
    verbose_text : boolean
        If eigenvectors should be printed to screen.

    """
    nc = 2*norb if spinpol else norb
    nb = n//(2*norb) - 1
    # If the diagonal Hamiltonian is due to the use of cubic harmonics
    cubic_basis = (basis_tag[-2:] == '03' or basis_tag[-6:] == '03-obs')

    # Obtain eigenvectors used to rotate from spherical harmonics to the
    # transformed basis.
    if cubic_basis:
        vtr = unitarytransform.get_spherical_2_cubic_matrix(spinpol, (norb-1)/2)
    elif irr_flag:
        vtr = np.zeros((nc, nc), dtype=np.complex)
        filename = 'proj-' + basis_tag + '-' + irr_flag + '.inp'
        print("Read rotation transformation from file:", filename)
        with open(filename, 'r') as fr:
            content = fr.readlines()
        for row in content[1:]:
            lst = row.split()
            if lst :
                vtr[int(lst[0])-1, int(lst[1])-1] = ( float(lst[2])
                                                     +1j*float(lst[3]))
    else:
        # No unitary transformation needed
        # Spherical harmonics to spherical harmonics
        vtr = np.eye(nc, dtype=np.complex)

    if verbose_text:
        print("Eigenvectors read from file:")
        print("Real part:")
        print(print_matrix(vtr.real))
        print("Imag part:")
        print(print_matrix(vtr.imag))

    # Construct unitary rotation matrix
    utr = np.transpose(np.conj(vtr))
    u = np.zeros((n,n), dtype=np.complex)
    # Bath orbitals
    if spherical_bath_basis:
        if spinpol:
            for i in range(nb):
                u[nc*(1+i):nc*(1+i+1), nc*(1+i):nc*(1+i+1)] = utr
        else:
            for i in range(nb):
                u[2*nc*(1+i):2*nc*(1+i)+nc, 2*nc*(1+i):2*nc*(1+i)+nc] = utr
                u[2*nc*(1+i)+nc:2*nc*(2+i), 2*nc*(1+i)+nc:2*nc*(2+i)] = utr
    else:
        for i in range(2*norb,2*norb*(1 + nb)):
            u[i, i] = 1
    # Impurity orbitals
    if spinpol:
        u[:nc, :nc] = utr
    else:
        u[:nc, :nc] = utr
        u[nc:2*nc, nc:2*nc] = utr
    return u
