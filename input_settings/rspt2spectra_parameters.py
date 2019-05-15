#!/usr/bin/env python3

"""

rspt2spectra_parameters
=======================

This file contains material specific information needed by the finiteH0.py
script. Put this file in the RSPt simulation folder.
Then execute the finiteH0.py script (which will read this file).

"""

import numpy as np
from rspt2spectra.constants import eV
from rspt2spectra.hybridization import get_wborders


# Verbose parameters. True or False
verbose_fig = True
verbose_text = True

# The non-relativistic non-interacting Hamiltonian operator
# is printed to this file name
output_filename = 'h0.pickle'

# Distance above real axis.
# This has to be the same as the RSPt value stored in green.inp.
eim = 0.005*eV

# Correlated orbitals to study, e.g.:
# '0102010100' or
# '0102010103' or
# '0102010103-obs' or
# '0102010100-obs1' or
# '0102010100-obs'
basis_tag = '0102010100-obs1'

# User defined local basis keyword.
# If not used any user defined local basis by
# providing a projection file, use an empty string: ''
irr_flag = 'Irr05'

# If to analyze spin-polarized calculations
spinpol = False

# If to analyze spin averaged calculations.
# This means spin-polarization, if any, is only from
# the hybridization and the self-energy
# If spinpol = False, this variable has no purpose.
spinavg = True

# Specify either initial bath energies or a energy window for each bath energy.
# It probably will be slightly more convienient for the impurityModel script
# to read the output from finiteH0.py if the bath sets here are ordered such
# that unoccupied bath states (e.i. those with positive energy) are put/stored
# after the occupied bath sets.
# Initial bath energies. One row contains the bath energies corresponding to
# one impurity orbital.
# Example with one (1+0) bath orbital per impurity orbital
#eb_initial = np.array([[-3],
#                       [-6],
#                       [-3],
#                       [-6],
#                       [-5]],dtype=np.float)
# Example with two (1+1) bath orbitals per impurity orbital
#eb_initial = np.array([[-3,2],
#                       [-6,4],
#                       [-3,2],
#                       [-6,4],
#                       [-5,3]],dtype=np.float)
# Energy windows for bath energies. One row contains the energies corresponding
# to one impurity orbital.
# Example with one (1+0)bath orbital per impurity orbital
#wborder = np.array([[[-8,-1]],
#                    [[-8,-1]],
#                    [[-8,-1]],
#                    [[-8,-1]],
#                    [[-8,-1]]],dtype=np.float)
# Example with two (2+0) bath orbitals per impurity orbital
#wborder = np.array([[[-8,-2.5], [-2.5,-1]],
#                    [[-8,-4],   [-4,-1]],
#                    [[-8,-2.5], [-2.5,-1]],
#                    [[-8,-4],   [-4,-1]],
#                    [[-8,-4],   [-4,-1]]],dtype=np.float)
# Example with two (1+1) bath orbitals per impurity orbital
#wborder = np.array([[[-8,0], [0,1]],
#                    [[-8,0], [0,1]],
#                    [[-8,0], [0,1]],
#                    [[-8,0], [0,1]],
#                    [[-8,0], [0,1]]],dtype=np.float)
# Example with three (3+0) bath orbitals per impurity orbital
#wborder = np.array([[[-8,-4],[-4,-2.5],[-2.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1]]],dtype=np.float)
# Example with three (2+1) bath orbitals per impurity orbital
#wborder = np.array([[[-8,-2.5],[-2.5,-1],[0,1]],
#                    [[-8,-4],[-4,-1],[0,1]],
#                    [[-8,-2.5],[-2.5,-1],[0,1]],
#                    [[-8,-4],[-4,-1],[0,1]],
#                    [[-8,-4],[-4,-1],[0,1]]],dtype=np.float)
# Example with four (4+0) bath orbitals per impurity orbital
#wborder = np.array([[[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1]]],dtype=np.float)
# Example with four (3+1) bath orbitals per impurity orbital
#wborder = np.array([[[-8,-4],[-4,-2.5],[-2.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1],[0,1]]],dtype=np.float)
# Example with four (4+1) bath orbitals per impurity orbital
#wborder = np.array([[[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1],[0,1]],
#                    [[-8,-4],[-4,-2.5],[-2.5,-1.5],[-1.5,-1],[0,1]],
#                    [[-8,-5],[-5,-2.5],[-2.5,-1.5],[-1.5,-1],[0,1]]],dtype=np.float)
# Example with ten (10+0) bath orbitals per impurity orbital
wborder = get_wborders(n_val=10, n_con=0)
# Example with eleven (10+1) bath orbitals per impurity orbital
#wborder = get_wborders(n_val=10, n_con=1)
# Example with twelve (10+2) bath orbitals per impurity orbital
#wborder = get_wborders(n_val=10, n_con=2)
# Example with twenty (20+0) bath orbitals per impurity orbital
#wborder = get_wborders(n_val=20, n_con=0)
# Example with thirty (30+0) bath orbitals per impurity orbital
#wborder = get_wborders(n_val=30, n_con=0)

# Plot energy range. Only used for plotting.
xlim = [-9,4]

# Method of choice for calculating on-site energies.
# Three possibilities:
# 0 - Considers non-interacting PDOS and neglects
#     off-diagonal elements of RSPt's hybridization
#     function.
# 1 - Considers non-interacting PDOS and includes
#     off-diagonal elements of RSPt's hybridization
#     function.
# 2 - Considers interacting PDOS and includes
#     off-diagonal elements of RSPt's
#     hybridization function.
e_method = 0

# If off-diagonal hybridization elements are
# available in files
off_diag_hyb = False

# If self-energy is available in files
self_energy = False

# Energy intervall where to search for solution
# of adjusted on-site energies.
bounds = (-3,0.5)

# Energy window in which the center of gravity
# of non-interacting PDOS is calculated.
# Choose range by inspecting RSPt's
# non-interacting PDOS.
wmin0 = -3
wmax0 = 2

# Energy window in which the center of gravity
# of interacting PDOS is calculated.
# Choose range by inspecting RSPt's
# interacting PDOS.
wmin = -8
wmax = 3

# If bath orbitals should be spherical harmonics or kept in rotated basis.
spherical_bath_basis = True

# If save Hamiltonian to Quanty friendly format
save2Quanty = True
