#!/usr/bin/env python3

"""

h2Quanty
======================

This module contains functions which are useful for interfacing to Quanty.

"""

import numpy as np
import scipy.sparse

# Write Hamiltonian in Quanty format
# - In Quanty, the orbital ordering is differently than in RSPt.
# A reordering is nececcary.
#
# - The Hamiltonian processes are split up into several groups.
# This is done because the .lua file is not read properly by
# Quanty if the operator contains "too" many elements.
# It seems the column width is somehow limited in the .lua files
# (when read and used by Quanty).
#
# - Due to the eventual core states (e.g. 2p), one has to be careful
# about the indices and shift the indices.

def print_QuantyH_from_dense_rsptH(h_rspt, ang=2, previous_orbitals=6,
                                   next_orbitals=0, filename='h0.lua',
                                   op_name='H0'):
    """
    Print Hamiltonian in a Quanty friendly format.

    Parameters
    ----------
    h_rspt : (N,N) array
        Hamiltonian matrix ordered in RSPt format.
        Contains impurity orbitals of one angular momentum and
        associated bath orbitals.
    ang : int
        Angular momentum of impurity orbitals.
    previous_orbitals : int
        Number of spin-orbitals appearing before current orbitals in
        the total Hamiltonian.
    next_orbitals : int
        Number of spin-orbitals appearing after current orbitals in
        the total Hamiltonian.
    filename : str
        Output filename.
    op_name : str
        Variable name for saved operator.

    """
    # Convert Hamiltonian to Quanty's orbital ordering format
    h_quanty = rspt_2_quanty_matrix(h_rspt,ang)
    norb = 2*ang + 1
    nb = np.shape(h_rspt)[0]//(2*norb) - 1
    # Number of spin-orbitals in the system studied in Quanty
    nf = previous_orbitals + 2*norb*(1 + nb) + next_orbitals
    # Index shifts
    ishift, jshift = previous_orbitals, previous_orbitals
    # Maximum number of elements in each operator
    nmax = 85
    # Transform matrix into several Quanty operators.
    # Save to disk.
    write_quanty_opps(h_quanty, nf, n=nmax, op_name=op_name, ishift=ishift,
                      jshift=jshift,filename=filename)
    # Save also all elements into one operator
    save21 = False
    if save21:
        # Transform matrix into one Quanty operator.
        # Save to disk.
        write_quanty_opp(h_quanty, nf, op_name='H_0', ishift=ishift,
                         jshift=jshift,filename='H0.lua')


def quanty_index(i,ang=2):
    """
    Return spin-orbital index in common Quanty ordering notation.

    In RSPt, spin-orbitals are ordered:
    ((orb1_dn, orb2_dn,   ... | orb1_up, orb2_up, ...), (... | ...), )

    In Quanty, a common notation is the spin-orbital ordering:
    (orb1_dn, orb1_up, orb2_dn, orb2_up, ...)

    Parameters
    ----------
    i : int
        index of orbital in RSPt ordering.
    ang : int
        Angular momentum of impurity orbitals.
    """
    norb = 2*ang + 1
    k = (i//(2*norb))*(2*norb)
    if (i-k) < norb:
        j = k + 2*(i-k)
    else:
        j = k + 2*((i-k)-norb) + 1
    return j

def rspt_2_quanty_matrix(x,ang=2):
    r"""
    Maps matrix from represented in the RSPt notation to common Quanty notation.

    In RSPt, spin-orbitals are ordered:
    ((orb1_dn, orb2_dn,   ... | orb1_up, orb2_up, ...), (... | ...), )

    In Quanty, a common notation is the spin-orbital ordering:
    (orb1_dn, orb1_up, orb2_dn, orb2_up, ...)

    Parameters
    ----------
    x : (N,N) array
        Matrix to transform.
    ang : int
        Angular momentum of impurity orbitals.

    Returns
    -------
    xt : matrix
        Matrix x, transformed into common Quanty notation.

    """
    xt = np.zeros_like(x)
    norb = 2*ang + 1
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            if x[i,j] != 0:
                iq = quanty_index(i,ang)
                jq = quanty_index(j,ang)
                xt[iq, jq] = x[i,j]
    return xt


def write_quanty_opp(x, nf, op_name='Opp', ishift=0, jshift=0,
                     filename='tmp.lua', mode='a'):
    r"""
    Write matrix to disk, as one Quanty operator.

    Parameters
    ----------
    x : (N,N) array
        Matrix to write to disk.
    nf : int
        Number of spin-orbitals in the system studied by Quanty.
    op_name : str
        Name of the operator.
    ishift : int
        Shift of row index to apply operator to correct spin-orbitals
        in the Quanty script.
    jshift : int
        Shift of column index to apply operator to correct spin-orbitals
        in the Quanty script.
    filename : str
        File where the matrix is saved.
    mode : {'a', 'w'}
        To append or overwrite.

    """

    if not (mode == 'a' or mode == 'w'):
        print('Warning: writing mode not supported.')
        return

    # create the three lists needed in Quanty: i,j,e
    i, j, e = scipy.sparse.find(x)

    # take care that indices may be shifted
    i += ishift
    j += jshift

    s = "\n"
    s += op_name
    s += " = NewOperator(\"Number\","
    s += str(nf)
    s += ",\n {"
    s += ', '.join([str(el) for el in i])
    s += '},\n {'
    s += ', '.join([str(el) for el in j])
    s += '},\n'
    s_onerow = '{'
    s_onerow += ', '.join([(str(el.real) + '+I*'
                            + str(el.imag)) for el in e])
    s_onerow += '})'
    s += s_onerow
    s += '\n'
    if len(s_onerow) > 3000:
        print('Warning: Lua does not support '
              'lines longer than 3000')
    f = open(filename, mode)
    f.write(s)
    f.close()


def write_quanty_opps(x, nf, n=1, op_name='Opp', ishift=0, jshift=0,
                      filename='tmp.lua', mode='a'):
    """
    Write matrix x to disk, by dividing it into
    several Quanty operators.

    Parameters
    ----------
    x : (N,N) array
        Dense matrix to be written in sparse format for Quanty.
    nf : int
        Number of spin-orbitals in the system studied by Quanty.
    n : int
        Max number of elements in each operator.
    op_name : str
        Common name of the operators.
    ishift : int
        To apply operator to correct spin-orbitals.
    jshift : int
        To apply operator to correct spin-orbitals.
    filename : str
        File where the matrix is saved.
    mode : {'a', 'w'}
        To append or overwrite.

    """

    if not (mode == 'a' or mode == 'w'):
        print('Warning: writing mode not supported.')
        return

    # create the three lists needed in Quanty: i,j,e
    i, j, e = scipy.sparse.find(x)

    # take care that indices may be shifted
    i += ishift
    j += jshift

    # divide up the elements in bunches of
    # max n in each bunch
    k = 0
    d = []
    while k + n < len(i):
        d.append([i[k:k + n], j[k:k + n], e[k:k + n]])
        k += n
    d.append([i[k:], j[k:], e[k:]])

    # write operators to file
    f = open(filename, mode)
    # loop over the operators
    for k, (ii, jj, ee) in enumerate(d):
        s = "\n"
        s += op_name + '_' + str(k)
        s += " = NewOperator(\"Number\","
        s += str(nf)
        s += ",\n {"
        s += ', '.join([str(el) for el in ii])
        s += '},\n {'
        s += ', '.join([str(el) for el in jj])
        s += '},\n'
        s_onerow = '{'
        s_onerow += ', '.join([(str(el.real) + '+I*'
                                + str(el.imag)) for el in ee])
        s_onerow += '})'
        s += s_onerow
        s += '\n'
        if len(s_onerow) > 3000:
            print('Warning: Lua does not support '
                  'lines longer than 3000')
        f.write(s)
    f.close()
