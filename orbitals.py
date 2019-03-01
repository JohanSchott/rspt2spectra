#!/usr/bin/env python3

"""

orbitals
========

This module contains functions which are relevant for the choice of localized orbitals.

"""

import numpy as np
import sys


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
