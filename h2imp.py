#!/usr/bin/env python3

"""

h2imp
=====

This module contains functions which are useful for interfacing to
the impurityModel repository.

"""

import numpy as np
import scipy.sparse
import pickle
from collections import OrderedDict

def write_to_file(d, filename='h0_Op.pickle'):
    """
    Write variable to disk.
    """
    with open(filename, 'wb') as handle:
        pickle.dump(d, handle)


def get_H_operator_from_dense_rspt_H_matrix(h, ang=2):
    """
    Returns Hamiltonian in a second quantization operator format.

    Parameters
    ----------
    h : (N,N) array
        Hamiltonian matrix ordered in RSPt format.
        Contains impurity orbitals of one angular momentum and
        associated bath orbitals.
        Spherical harmonics basis is assumed for impurity orbitals.
    ang : int
        Angular momentum of impurity orbitals.

    Returns
    -------
    hOp : dict
        tuple : complex,
        where the key-tuple describe the physical process of an electron
        annihilation at spin orbital j, followed by an electron creation
        at spin orbital i.
    """

    assert np.shape(h)[0] == np.shape(h)[1]
    # Number of spin orbitals in total.
    n = np.shape(h)[0]
    # Specify how many bath states are present.
    nBaths = OrderedDict()
    nBaths[ang] = n - 2*(2*ang+1)
    hOp = {}
    for i in range(n):
        for j in range(n):
            if h[i,j] != 0:
                spin_orb_i = rspt_i2c(nBaths, i)
                spin_orb_j = rspt_i2c(nBaths, j)
                hOp[((spin_orb_i, 'c'), (spin_orb_j, 'a'))] = h[i,j]
    return hOp

def rspt_i2c(nBaths, i):
    """
    Return an coordinate tuple, representing a spin-orbital.

    Parameters
    ----------
    nBaths : ordered dict
        An elements is either of the form:
        angular momentum : number of bath spin-orbitals
        or of the form:
        (angular momentum_a, angular momentum_b, ...) : number of bath states.
        The latter form is used if impurity orbitals from different
        angular momenta share the same bath states.
    i : int
        An index denoting a spin-orbital or a bath state.

    Returns
    -------
    spinOrb : tuple
        (l, s, m), (l, b) or ((l_a, l_b, ...), b)

    """
    # Counting index.
    k = 0
    # Check if index "i" belong to an impurity spin-orbital.
    # Loop through all impurity spin-orbitals.
    for lp in nBaths.keys():
        if isinstance(lp, int):
            # Check if index "i" belong to impurity spin-orbital having lp.
            if i - k < 2*(2*lp+1):
                for sp in range(2):
                    for mp in range(-lp, lp+1):
                        if k == i:
                            return (lp, sp, mp)
                        k += 1
            k += 2*(2*lp+1)
        elif isinstance(lp, tuple):
            # Loop over all different angular momenta in lp.
            for lp_int in lp:
                # Check if index "i" belong to impurity spin-orbital having lp_int.
                if i - k < 2*(2*lp_int+1):
                    for sp in range(2):
                        for mp in range(-lp_int, lp_int+1):
                            if k == i:
                                return (lp_int, sp, mp)
                            k += 1
                k += 2*(2*lp_int+1)
    # If reach this point it means index "i" belong to a bath state.
    # Need to figure out which one.
    for lp, nBath in nBaths.items():
        b = i - k
        # Check if bath state belong to bath states having lp.
        if b < nBath:
            # The index "b" will have a value between 0 and nBath-1
            return (lp, b)
        k += nBath
    print(i)
    sys.exit('Can not find spin-orbital state corresponding to index.')
