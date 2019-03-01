#!/usr/bin/env python3


# Diagonalize 3d-block of Hamiltonian $H_0$
# Find basis (eigenvectors) which diagonalizes non-interacting local Hamiltonian $H_0$.

# Choose simulation folder to study.

# The script will read `Local hamiltonian` from RSPt `out` file.
# Then, diagonalize Hamiltonian and use eigenvectors to transform from spherical harmonics basis
# to basis where Hamiltonian is diagonal.
# Finally the script will save transformation vectors in RSPt-friendly format to disk.


import numpy as np
import sys

from rspt2spectra import energies, orbitals


def main():

    if len(sys.argv) > 1 and sys.argv[1] == 'spinpol':
        # RSPt calculations are spin-polarized
        spinpol = True
    else:
        spinpol = False

    # Label for user defined transformation matrix.
    # E.g. name of irreducible representation.
    if spinpol and len(sys.argv) == 3:
        irrlabel = sys.argv[2]
    elif not spinpol and len(sys.argv) == 2:
        irrlabel = sys.argv[1]
    else:
        irrlabel = '1'

    hs, labels = energies.parse_matrices('out')
    # Loop over all the local Hamiltonians
    for clusterIndex, (h, label) in enumerate(zip(hs, labels)):
        print('------ Cluster label:', label, '------')
        # Number of correlated orbitals
        # 5 for d-orbitals, 7 for f-orbitals
        norb = np.shape(h)[0]//2
        # Name of the file to write into
        fwrite = 'proj-' + label + '-Irr' + irrlabel + '.inp'
        if spinpol:
            hd = np.copy(h)
        else:
            hd = h[0:norb, 0:norb]  # dn block
        print('Hamiltonian:')
        print('Real part:')
        print(energies.print_matrix(hd.real))
        print('Imag part:')
        print(energies.print_matrix(hd.imag))
        print()
        # Eigenvalues and eigenvectors (column vectors)
        e, v = np.linalg.eigh(hd)
        # Reorder eigenvectors
        if spinpol:
            vs = np.zeros_like(v)
            es = np.zeros_like(e)
            a, b = 0, norb
            for i in range(2*norb):
                if np.sum(np.abs(v[:norb, i])**2) > 0.5:
                    vs[:, a] = v[:, i]
                    es[a] = e[i]
                    a += 1
                else:
                    vs[:, b] = v[:, i]
                    es[b] = e[i]
                    b += 1
        else:
            vs = np.copy(v)
            es = np.copy(e)
        print("Eigenvalues:")
        print(es)
        if spinpol:
            print('Eigenvalues: e(dn)-e(up):')
            print(es[:5]-es[5:])
        print("Eigenvectors:")
        print('Real part:')
        print(energies.print_matrix(vs.real))
        print('Imag part:')
        print(energies.print_matrix(vs.imag))
        print('Diagonalized Hamiltonian:')
        hdiag = np.dot(np.transpose(np.conj(vs)), np.dot(hd, vs))
        print('Real part:')
        print(energies.print_matrix(hdiag.real))
        print('Imag part:')
        print(energies.print_matrix(hdiag.imag))
        print()
        # Save rotation matrices to file in a RSPt adapted format
        orbitals.write_proj_file(vs, spinpol, fwrite)


if __name__ == "__main__":
    main()
