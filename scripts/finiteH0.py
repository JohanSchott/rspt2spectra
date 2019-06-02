#!/usr/bin/env python3

# finiteH0
# Script for analyzing hybridization functions generated by RSPt.
# Generate a finite size non-interacting Hamiltonian, expressed in
# a spherical harmonics basis.


import matplotlib.pylab as plt
import numpy as np

from rspt2spectra.constants import eV
from rspt2spectra import readfile
from rspt2spectra import offdiagonal
from rspt2spectra import energies
from rspt2spectra import h2imp
# Read input parameters from local file
import rspt2spectra_parameters as r2s

# We need to specify how many bath sets should be used for each block.
assert np.shape(r2s.n_bath_sets_foreach_block_and_window)[0] == len(r2s.blocks)
# We need to specify how many bath sets should be used for each window.
assert np.shape(r2s.n_bath_sets_foreach_block_and_window)[1] == len(r2s.wborders)

# Help variables
# Files for hybridization function
file_re_hyb = 'real-hyb-' + r2s.basis_tag + '.dat'
file_im_hyb = 'imag-hyb-' + r2s.basis_tag + '.dat'
# Name of RSPt's output file.
outfile = 'out'
# Number of considered impurity orbitals
n_imp = sum(len(block) for block in r2s.blocks)

# Read RSPt's diagonal hybridization functions
w, hyb_diagonal = readfile.hyb(file_re_hyb, file_im_hyb,
                               only_diagonal_part=True)
assert n_imp == np.shape(hyb_diagonal)[0]
print(np.shape(hyb_diagonal))
# All hybridization functions
w, hyb = readfile.hyb(file_re_hyb, file_im_hyb)
print(np.shape(hyb))

if r2s.verbose_fig:
    # Plot diagonal and off diagonal hybridization functions separately.
    offdiagonal.plot_diagonal_and_offdiagonal(w, hyb_diagonal, hyb, r2s.xlim)
    # Plot all orbitals, both real and imaginary parts.
    offdiagonal.plot_all_orbitals(w, hyb, xlim=r2s.xlim)

# Calculate bath and hopping parameters.
eb, v = offdiagonal.get_eb_v(w, r2s.eim, hyb, r2s.blocks, r2s.wsparse,
                             r2s.wborders,
                             r2s.n_bath_sets_foreach_block_and_window,
                             r2s.xlim, r2s.verbose_fig, r2s.gamma)

print('\n \n')
print('Bath state energies')
print(np.array_str(eb, max_line_width=1000, precision=3, suppress_small=True))
print('Hopping parameters')
print(np.array_str(v, max_line_width=1000, precision=3, suppress_small=True))
print('Shape of bath state energies:', np.shape(eb))
print('Shape of hopping parameters:', np.shape(v))

if r2s.verbose_fig:
    # Relative distribution of hopping parameters
    plt.figure()
    plt.hist(np.abs(v).flatten()/np.max(np.abs(v)),bins=100)
    plt.xlabel('|v|/max(|v|)')
    plt.show()
    # Relative values of the hopping parameters
    plt.figure()
    plt.plot(sorted(np.abs(v).flatten())/np.max(np.abs(v)),'-o')
    plt.ylabel('|v|/max(|v|)')
    plt.show()

    # Distribution of hopping parameters
    plt.figure()
    plt.hist(np.abs(v).flatten(),bins=100)
    plt.xlabel('|v|')
    plt.show()
    # Absolute values of the hopping parameters
    plt.figure()
    plt.plot(sorted(np.abs(v).flatten()),'-o')
    plt.ylabel('|v|')
    plt.show()

print('{:d} elements in v.'.format(v.size))
v_mean = np.mean(np.abs(v))
v_median = np.median(np.abs(v))
print('<v> = ', v_mean)
print('v_median = ', v_median)
r_cutoff = 0.01
mask = np.abs(v) < r_cutoff*np.max(np.abs(v))
print('{:d} elements in v are smaller than {:.3f}*v_max.'.format(
    v[mask].size, r_cutoff))

# Check small non-zero values.
mask = np.logical_and(0 < np.abs(v), np.abs(v) < r_cutoff*np.max(np.abs(v)))
print('{:d} elements in v are close to zero (of {:d})'.format(
    v[mask].size, v.size))
# One might want to put these hopping parameters to zero.
#v[mask] = 0

#print('Absolut values of these elements:')
#print(sorted(np.abs(v[mask])))


# Extract the impurity energies from the local Hamiltonian
# and the chemical potential.
hs, labels = energies.parse_matrices(outfile)
mu = energies.get_mu()
for h, label in zip(hs, labels):
    # Select Hamiltonian from correct cluster
    if label == r2s.basis_tag:
        print("Extract local H0 from cluster:", label)
        print()
        e_rspt = eV*(h - mu*np.eye(n_imp))
        #e_rspt = eV*np.real(h.diagonal() - mu)

print("RSPt's local hamiltonian")
print(np.array_str(e_rspt, max_line_width=1000, precision=3,
                   suppress_small=True))
print()
print("Eigenvalues of RSPt's local Hamiltonian:")
eig, _ = np.linalg.eigh(e_rspt)
print(eig)
print()

# Construct the non-interacting Hamiltonian
h = np.zeros((n_imp+len(eb),n_imp+len(eb)), dtype=np.complex)
# Onsite energies of impurity orbitals
h[:n_imp,:n_imp] = e_rspt
# Bath state energies
np.fill_diagonal(h[n_imp:, n_imp:], eb)
# Hopping parameters
h[n_imp:,:n_imp] = v
h[:n_imp,n_imp:] = np.conj(v).T


# Make sure Hamiltonian is hermitian
assert np.sum(np.abs(h - np.conj(h.T))) < 1e-12

if r2s.verbose_text:
    print("Dimensions of Hamiltonian:", np.shape(h))
    print("Hamiltonian in spherical harmonics basis:")
    print("Correlated block:")
    print("Real part:")
    print(np.array_str(np.real(h[:n_imp, :n_imp]), precision=3,
                       suppress_small=True))
    print('Imag part:')
    print(np.array_str(np.imag(h[:n_imp, :n_imp]), precision=3,
                       suppress_small=True))
    print("Number of non-zero elements in H:",len(np.flatnonzero(h)))

hOperator = h2imp.get_H_operator_from_dense_rspt_H_matrix(h,
                                                          ang=(n_imp//2-1)//2)
if r2s.verbose_text:
    print("Hamiltonian operator:")
    print(hOperator)
    #repr(hOperator)
    print()
    print('len(hOperator) = {:d}'.format(len(hOperator)))
    print('{:.3f} bath states per impurity spin-orbital.'.format(
        (np.shape(h)[0] - n_imp)/n_imp))
    print('{:d} bath states in total.'.format(np.shape(h)[0] - n_imp))
h2imp.write_to_file(hOperator, r2s.output_filename)
