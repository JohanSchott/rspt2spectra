#!/usr/bin/env python3

"""

rspt2spectra_parameters
=======================

This file contains material specific information needed by
the finiteH0.py script. Put this file in the RSPt simulation folder.
Then execute the finiteH0.py script (which will read this file).

"""

import numpy as np
from rspt2spectra.constants import eV


# Correlated orbitals to study.
# Examples:
# '0102010100' or
# '0102010103' or
# '0102010103-obs' or
# '0102010100-obs1' or
# '0102010100-obs'
basis_tag = '0102010100'

# User defined local basis keyword.
# If not used any user defined local basis by
# providing a projection file, use an empty string: ''
# Examples:
#irr_flag = 'Irr05'
#irr_flag = ''
irr_flag = ''

# Distance above real axis.
# This has to be the same as the RSPt value stored in green.inp.
eim = 0.005*eV

# Verbose parameters. True or False
verbose_fig = True
verbose_text = True

# Plot energy range. Only used for plotting.
xlim = (-9, 4)

# The non-relativistic non-interacting Hamiltonian operator
# is printed to this file name
output_filename = 'h0.pickle'

# Window borders. Divide up the energy mesh in regions.
# Only one wide region also works fine.
# Example with 1 window.
wborders = np.array([[-8, 0]], dtype=np.float)
# Example with 2 windows. One for occupied and one for unoccupied bath states.
#wborders = np.array([[-8, 0], [0, 2]], dtype=np.float)

# Input which impurity orbitals belong to each block.
# Note, first counting index is 0, not 1 (Python notation).
# Example with 6 blocks and 10 spin-orbitals:
#blocks = (np.array([0, 3]),
#          np.array([1, 4]),
#          np.array([2]),
#          np.array([5, 8]),
#          np.array([6, 9]),
#          np.array([7]))
# Example with 8 blocks and 10 spin-orbitals (e.g. 3d orbitals in spherical harmonics basis):
blocks = (np.array([0, 4]),
          np.array([1]),
          np.array([2]),
          np.array([3]),
          np.array([5, 9]),
          np.array([6]),
          np.array([7]),
          np.array([8]))
# Example with 10 blocks and 10 spin-orbitals (e.g. 3d orbitals in cubic harmonics basis):
#blocks = (np.array([0]),
#          np.array([1]),
#          np.array([2]),
#          np.array([3]),
#          np.array([4]),
#          np.array([5]),
#          np.array([6]),
#          np.array([7]),
#          np.array([8]),
#          np.array([9]))

# Number of bath sets for each block and energy window.
# Example compatible with 5 blocks and 1 window:
#n_bath_sets_foreach_block_and_window = np.array([[10], [10], [10], [10], [10]])
#n_bath_sets_foreach_block_and_window = np.array([[10], [10], [5], [5], [5]])
# Examples compatible with 6 blocks and 1 window:
#n_bath_sets_foreach_block_and_window = np.array([[4], [4], [4], [4], [4], [4]])
#n_bath_sets_foreach_block_and_window = np.array([[10], [10], [10], [10], [10], [10]])
#n_bath_sets_foreach_block_and_window = np.array([[20], [20], [20], [20], [20], [20]])
#n_bath_sets_foreach_block_and_window = np.array([[30], [30], [30], [30], [30], [30]])
# Examples compatible with 6 blocks and 2 windows:
#n_bath_sets_foreach_block_and_window = np.array([[10, 1], [10, 1], [10, 1], [10, 1], [10, 1], [10, 1]])
#n_bath_sets_foreach_block_and_window = np.array([[10, 2], [10, 2], [10, 2], [10, 2], [10, 2], [10, 2]])
#n_bath_sets_foreach_block_and_window = np.array([[30, 1], [30, 1], [30, 1], [30, 1], [30, 1], [30, 1]])
#n_bath_sets_foreach_block_and_window = np.array([[30, 1], [30, 1], [10, 1], [10, 1], [10, 1], [10, 1]])
# Examples compatible with 8 blocks and 1 window:
n_bath_sets_foreach_block_and_window = np.array([[30], [30], [30], [30], [30], [30], [30], [30]])
# Examples compatible with 10 blocks and 1 window:
#n_bath_sets_foreach_block_and_window = np.array([[30], [30], [30], [30], [30], [30], [30], [30], [30], [30]])

# Select how sparse the real mesh should be. 
# A value of 3 means that every third RSPt mesh-point is used.
wsparse = 2

# Regularization parameter. gamma = 0.01 is good.
gamma = 0.01

