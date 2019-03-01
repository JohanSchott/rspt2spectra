#!/usr/bin/env python3

"""

h2imp
=====

This module contains functions which are useful for interfacing to impurityModel.

"""

import numpy as np
import scipy.sparse
import pickle


def write_to_file(d, filename='h0_Op.pickle'):
    """
    Write variable to disk.
    """
    with open(filename, 'wb') as handle:
        pickle.dump(d, handle)


def get_H_operator_from_dense_rsptH(h, ang=2):
    """
    Returns Hamiltonian in a second quantization operator format.

    Parameters
    ----------
    h : (N,N) array
        Hamiltonian matrix ordered in RSPt format.
        Contains impurity orbitals of one angular momentum and
        associated bath orbitals.
        Spherical harmonics basis is assumed,
        for both impurity and bath orbitals.
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
    hOp = {}
    for i in range(np.shape(h)[0]):
        for j in range(np.shape(h)[1]):
            if h[i,j] != 0:
                spin_orb_i = index2coordinates(i,ang)
                spin_orb_j = index2coordinates(j,ang)
                hOp[((spin_orb_i,'c'),(spin_orb_j,'a'))] = h[i,j]
    return hOp

def index2coordinates(i,ang=2):
    """
    Return coordinates of orbital with Hamiltonian index i.

    Parameters
    ----------
    i : int
        Hamiltonian index.
    ang : int
        Angular momentum of impurity orbitals.


    Returns
    -------
    spin_orb : tuple
        (l,m,s) or (l,m,s,bath_set), where
        l = ang,
        m will take one value from {-l,-l+1,...,l},
        s will take one value from {0,1}, and
        bath_set will index which bath set the bath orbital belongs to,
        starting from 0.

    """
    norb = 2*ang+1
    k = i//(2*norb)
    i_imp = i-k*2*norb
    if i_imp < norb:
        s = 0
        m = i_imp - 2
    else:
        s = 1
        m = i_imp - 7
    if k == 0:
        return (ang,m,s)
    else:
        return (ang,m,s,k-1)
