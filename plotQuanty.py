#!/usr/bin/env python3

"""

plotQuanty
======================

This module contains functions which are useful to process data 
generated from the Quanty software.
The intension is to avoid material specific functions/variables 
in this module.

"""

import numpy as np
from math import pi
import scipy.sparse 

from constants import k_B

__author__ = "Johan Schott"
__email__ = "johan.schott@gmail.com"
__status__ = "Development"

# Number operators, in spherical harmonics basis.
n_3d = np.eye(5) 
n_2p = np.eye(6)

# Crystal-field operators and functions, in spherical harmonics basis.
# Dictonary of crystal-field matrices
d = {}
# Dq in spherical harmonics basis:
d['q'] = np.zeros((5,5)) 
np.fill_diagonal(d['q'],[1,-4,6,-4,1])
d['q'][-1,0] = 5
d['q'][0,-1] = 5
# Ds in spherical harmonics basis:
d['s'] = np.zeros((5,5)) 
np.fill_diagonal(d['s'],[2,-1,-2,-1,2])
# Dt in spherical harmonics basis:
d['t'] = np.zeros((5,5)) 
np.fill_diagonal(d['t'],[-1,4,-6,4,-1])

# Operators for another formulation 
# of the crystal-field Hamiltonian
delta = {}
delta['o'] = 1/10.*d['q']
delta['1'] = 1/20.*d['q']+1/7.*d['s']+3/35.*d['t']
delta['2'] = -1/15.*d['q']+1/7.*d['s']-4/35.*d['t']

def get_Dq_Ds_Dt(deltao, delta1, delta2):
    """
    Convert crystal-field parameters 
    from one representation to another.

    Parameters
    ----------
    deltao : float
        Energy separation between e_g and t_2g orbitals.
    delta1 : float
        Energy separation between x^2-y^2 and z^2 orbitals.
    delta2 : float
        Energy separation between xy and xz orbitals (xz same as yz).
    
    Returns
    -------
    dq : float
    ds : float
    dt : float

    """
    dq = 1/10.*deltao+1/60.*(3*delta1-4*delta2)
    ds = (delta1+delta2)/7.
    dt = (3*delta1-4*delta2)/35.
    return dq,ds,dt

def get_deltao_delta1_delta2(dq, ds, dt):
    '''
    Convert crystal-field parameters 
    from one representation to another.

    Parameters
    ----------
    dq : float
    ds : float
    dt : float
    
    Returns
    -------
    deltao : float
        Energy separation between e_g and t_2g orbitals.
    delta1 : float
        Energy separation between x^2-y^2 and z^2 orbitals.
    delta2 : float
        Energy separation between xy and xz orbitals (xz same as yz).

    '''
    deltao = 10*dq
    delta1 = 4*ds+5*dt
    delta2 = 3*ds-5*dt
    return deltao, delta1, delta2

def get_CF_hamiltonian_with_Dopp(e_mean,dq,ds,dt):
    '''
    Returns crystal-field hamiltonian in 
    spherical harmonics basis.

    '''
    h = e_mean*n_3d + dq*d['q'] + ds*d['s'] + dt*d['t']
    return h

def get_CF_hamiltonian(e_mean,deltao,delta1,delta2):
    '''
    Returns crystal-field hamiltonian in 
    spherical harmonics basis.

    '''
    h = (e_mean*n_3d + deltao*delta['o'] + 
         delta1*delta['1'] + delta2*delta['2'])
    # Alternative parameterization (giving the same matrix)
    #dq,ds,dt = get_Dq_Ds_Dt(deltao, delta1, delta2)
    #h = e_mean*n_3d + dq*d['q'] + ds*d['s'] + dt*d['t']
    return h


# Hybridization operators and functions, in spherical harmonics basis.
# The crystal-field Hamiltonian with one set of 
# crystal ligand orbitals:
# \begin{equation}
# h =
# \begin{bmatrix}
#     0  & & h_\mathrm{hyb}^\dagger \\
#     h_\mathrm{hyb}  & 0  
# \end{bmatrix}
# \end{equation},
# where the first diagonal block is for the 3d-orbitals and 
# the other diagonal is for the bath orbitals.
# Dictonary of different matrices.
hyb = {}
hyb['eg'] = np.zeros((5,5))
np.fill_diagonal(hyb['eg'],[0.5,0,1,0,0.5])
hyb['eg'][0,-1] = 0.5
hyb['eg'][-1,0] = 0.5
hyb['t2g'] = np.zeros((5,5))
np.fill_diagonal(hyb['t2g'],[0.5,1,0,1,0.5])
hyb['t2g'][0,-1] = -0.5
hyb['t2g'][-1,0] = -0.5
hyb['z2'] = np.zeros((5,5))
np.fill_diagonal(hyb['z2'],[0,0,1,0,0])
hyb['x2y2'] = np.zeros((5,5))
np.fill_diagonal(hyb['x2y2'],[0.5,0,0,0,0.5])
hyb['x2y2'][0,-1] = 0.5
hyb['x2y2'][-1,0] = 0.5
hyb['xz_yz'] = np.zeros((5,5))
np.fill_diagonal(hyb['xz_yz'],[0,1,0,1,0])
hyb['xy'] = np.zeros((5,5))
np.fill_diagonal(hyb['xy'],[0.5,0,0,0,0.5])
hyb['xy'][0,-1] = -0.5 
hyb['xy'][-1,0] = -0.5

def get_h_hyb_Oh(veg,vt2g):
    '''
    Return hybridization matrix in Oh symmetry.
    '''
    h_hyb_eg = veg*hyb['eg']
    h_hyb_t2g = vt2g*hyb['t2g']
    h_hyb = h_hyb_eg + h_hyb_t2g 
    return h_hyb

def get_h_hyb(vz2,vx2y2,vxz_yz,vxy):
    '''
    Return hybridization matrix in D4h symmetry.
    '''
    h_hyb_z2 = vz2*hyb['z2']
    h_hyb_x2y2 = vx2y2*hyb['x2y2']
    h_hyb_xz_yz = vxz_yz*hyb['xz_yz']
    h_hyb_xy = vxy*hyb['xy']
    # Combine hybridization terms
    h_hyb = h_hyb_z2 + h_hyb_x2y2 + h_hyb_xz_yz + h_hyb_xy
    return h_hyb


# Spin-orbit coupling operators, in spherical harmonics basis.
# (Values are taken from Quanty)
ldots_2p = np.zeros((6,6)) 
ldots_2p[1,1] = -0.5
ldots_2p[0,0] = 0.5
ldots_2p[5,5] = 0.5
ldots_2p[4,4] = -0.5
ldots_2p[2,1] = 7.071067811865476E-01
ldots_2p[1,2] = 7.071067811865476E-01
ldots_2p[4,3] = 7.071067811865476E-01
ldots_2p[3,4] = 7.071067811865476E-01

# Dipole transition operators, in spherical harmonics basis.
# (Values are taken from Quanty)
t = {}
# x polarization
t['x'] = np.zeros((10,6)) 
np.fill_diagonal(t['x'],[4.472135954999579E-01,
                         4.472135954999579E-01,
                         3.162277660168379E-01,
                         3.162277660168379E-01,
                         1.825741858350554E-01,
                         1.825741858350554E-01])
t['x'][4,0] = -1.825741858350554E-01
t['x'][5,1] = -1.825741858350554E-01
t['x'][6,2] = -3.162277660168379E-01
t['x'][7,3] = -3.162277660168379E-01
t['x'][8,4] = -4.472135954999579E-01
t['x'][9,5] = -4.472135954999579E-01
# y polarization
t['y'] = np.zeros((10,6),dtype=np.complex) 
np.fill_diagonal(t['y'],[4.472135954999579E-01,
                         4.472135954999579E-01,
                         3.162277660168379E-01,
                         3.162277660168379E-01,
                         1.825741858350554E-01,
                         1.825741858350554E-01])
t['y'][4,0] = 1.825741858350554E-01
t['y'][5,1] = 1.825741858350554E-01
t['y'][6,2] = 3.162277660168379E-01
t['y'][7,3] = 3.162277660168379E-01
t['y'][8,4] = 4.472135954999579E-01
t['y'][9,5] = 4.472135954999579E-01
t['y'] *= 1j
# z polarization
t['z'] = np.zeros((10,6)) 
t['z'][2,0] = 4.472135954999579E-01
t['z'][3,1] = 4.472135954999579E-01
t['z'][4,2] = 5.163977794943222E-01
t['z'][5,3] = 5.163977794943222E-01
t['z'][6,4] = 4.472135954999579E-01
t['z'][7,5] = 4.472135954999579E-01
# right polarization
t['r'] = np.zeros((10,6))
np.fill_diagonal(t['r'],[6.324555320336759E-01,
                         6.324555320336759E-01,
                         4.472135954999580E-01,
                         4.472135954999580E-01,
                         2.581988897471611E-01,
                         2.581988897471611E-01])
# left polarization
t['l'] = np.zeros((10,6))
t['l'][4,0] = 2.581988897471611E-01
t['l'][5,1] = 2.581988897471611E-01
t['l'][6,2] = 4.472135954999580E-01
t['l'][7,3] = 4.472135954999580E-01
t['l'][8,4] = 6.324555320336759E-01
t['l'][9,5] = 6.324555320336759E-01

# isotropic. Use this matrix with care.
t['z_r_l'] = t['z'] + t['r'] + t['l']

def quanty_2_rspt(x):
    r"""
    Maps matrix from represented in the Quanty notation 
    to RSPt's notation.
    
    In Quanty, spin-orbitals are ordered: 
    orb1_dn, orb1_up, orb2_dn, orb2_up, ...

    In RSPt, spin-orbitals are ordered:   
    orb1_dn, orb2_dn,   ... | orb1_up, orb2_up, ...
    
    Parameters
    ----------
    x : matrix
        Matrix to transform.
    
    Returns
    -------
    xR : matrix
        Matrix x, transformed into RSPt's notation.

    """
    n,m = np.shape(x)
    xR = np.zeros_like(x)
    if n%2 != 0 or m%2 != 0:
        print('Warning: matrix should have even dimensions.')
    for i in range(n):
        for j in range(m):
            if i%2 == 0:
                iR = i/2
            else:
                iR = n/2 + i/2
            if j%2 == 0:
                jR = j/2
            else:
                jR = m/2 + j/2
            xR[iR,jR] = x[i,j]
    return xR


def write_quanty_opp(x,nf,op_name='Opp',ishift=0,jshift=0,
                     filename='tmp.lua',mode='a'):
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
    i,j,e = scipy.sparse.find(x)
    
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
    f = open(filename,mode) 
    f.write(s)
    f.close() 

def write_quanty_opps(x,nf,n=1,op_name='Opp',ishift=0,jshift=0,
                      filename='tmp.lua',mode='a'):
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
    i,j,e = scipy.sparse.find(x)
    
    # take care that indices may be shifted
    i += ishift
    j += jshift
 
    # divide up the elements in bunches of
    # max n in each bunch
    k = 0
    d = []
    while k+n < len(i): 
        d.append([i[k:k+n],j[k:k+n],e[k:k+n]])
        k += n 
    d.append([i[k:],j[k:],e[k:]])
    
    # write operators to file
    f = open(filename,mode) 
    # loop over the operators
    for k,(ii,jj,ee) in enumerate(d):
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

# Reading and processing XA spectra, generated by Quanty

def fermi_smearing(x,y,fwhm):
    r"""
    Return Fermi smeared values.
    
    Smearing is done by convolution of the input data
    with the function:

    .. math:: f(x) = -\frac{\partial n(x)}{\partial x}, 
    
    where 
    
    .. math:: n(x) = \frac{1}{\exp{(\beta x)}+1}
    
    is the Fermi-dirac distribution.
    
    Assumes uniform grid.

    Parameters
    ----------    
    x : array
    y : array
    fwhm : float
        Full Width Half Max.

    """
    # Inverse temperature
    beta = 4.*np.log(1+np.sqrt(2))/fwhm
    # Grid spacing (assume uniform grid)
    dx = x[1]-x[0]
    # Grid spacing should be smaller than the 
    # standard deviation
    assert dx < fwhm
    # Create a mesh for the smearing function
    xg = np.arange(-3*fwhm,3*fwhm,dx)
    # Create smear function 
    g = beta*np.exp(beta*xg)/(np.exp(beta*xg)+1)**2
    # This check ensures the return array has the 
    # same shape as x
    assert len(x) > len(xg)
    # Convolute y with smearing function
    smeared = dx*np.convolve(y,g,mode='same')
    return smeared

def gaussian_smearing(x,y,fwhm):
    '''
    Return Gaussian smeared values of variable y. 
    
    Assumes uniform grid.
    
    Parameters
    ----------    
    x - array
    y - array
    fwhm - float
        Full Width Half Max.

    '''
    # the standard deviation of the Gaussian
    std = 1./(2*np.sqrt(2*np.log(2)))*fwhm
    # grid spacing (assume uniform grid)
    dx = x[1]-x[0]
    # Grid spacing should be smaller than the 
    # standard deviation
    assert dx < std
    # Create a mesh for the Gaussian function
    xg = np.arange(-5*std,5*std,dx)
    # Create a gaussian centered around zero 
    g = 1./np.sqrt(2*pi*std**2)*np.exp(-xg**2/(2*std**2))
    # This check ensures the return array has the 
    # same shape as x
    assert len(x) > len(xg)
    # Convolute y with Gaussian
    smeared = dx*np.convolve(y,g,mode='same')
    return smeared

def lorentzian_smearing(x,y,fwhm):
    '''
    Return Lorenzian smeared values of variable y. 
    
    Assumes uniform grid.
    
    Parameters
    ----------    
    x - array
    y - array
    fwhm - float
        Full Width Half Max.
    
    '''
    # delta variable in the Lorentzian
    d = fwhm/2.
    # Grid spacing (assume uniform grid)
    dx = x[1]-x[0]
    # Grid spacing should be smaller than the 
    # standard deviation
    assert dx < d
    # Create a mesh for the Lorentzian function
    xg = np.arange(-10*d,10*d,dx)
    # Create a Lorentzian centered around zero 
    g = 1/pi*d/(xg**2+d**2)
    # This check ensures the return array has the 
    # same shape as x
    assert len(x) > len(xg)
    # Convolute y with Lorentzian
    smeared = dx*np.convolve(y,g,mode='same')
    return smeared
    
def read_data_file(filename):
    """
    Return the spectra read from file.

    """
    with open(filename) as f:
        content = f.readlines()
    save_line = False
    x = []
    for line in content:
        columns = line.split()
        if save_line:
            x.append([float(item) for item in columns])
        if columns[0] == 'Energy':
            save_line = True
    return np.array(x)

def read_output_file(filename):
    """
    Read the eigenstate information read from file.
    
    """
    with open(filename) as f:
        content = f.readlines()
    save_line = False
    x = []
    for line in content:
        columns = line.split()
        if save_line and columns:
            try:
                x.append([float(e) for e in columns])
            except ValueError:
                save_line = False
                break
        if len(columns) >= 2  and columns[1] == 'states':
            save_line = True
    return np.array(x)

def get_normalization(x,minus_intensity,loc,h,peak=0,
                      peakorder='topdown'):
    y = -minus_intensity
    i = find_peak_index(x,y,peak,peakorder)
    #print i
    shift = loc - x[i]
    #print x[i]
    scale = h/minus_intensity[i]
    return shift, scale
        
def find_peak_index(x,y,peak_nbr=0,peakorder='topdown'):
    r"""
    Return position index of the `peak_nbr` highest or 
    leftest peak. 

    Parameters
    ----------
    x : (N) array
        Energy mesh.
    y : (N) array
    peak_nbr : int
        Peak index.
    peakorder: {'topdown', 'leftright'}
        How to sort peaks in variable `y`.

    """
    if peakorder == 'topdown' and peak_nbr == 0:
        return np.argmax(y) # this is a special case
    else:    
        mask1 = np.logical_and(0<=np.diff(y[:-1]),
                               np.diff(y[1:])<0)
        mask2 = np.logical_and(0<np.diff(y[:-1]),
                               np.diff(y[1:])<=0)
        mask = np.logical_or(mask1,mask2)
        peak_x = x[1:-1][mask]
        peak_y = y[1:-1][mask]
        #print peak_x
        if peakorder == 'topdown':
            indices = np.argsort(peak_y)
            index = indices[-1-peak_nbr]
        elif peakorder == 'leftright':
            indices = np.argsort(peak_x)
            index = indices[peak_nbr]
        else:
            print('Warning: Value of peakorder variable'
                   'is incorrect.')
        return np.argmin(np.abs(x-peak_x[index])) 

def thermal_average(energies,observable,T=300):
    '''
    Returns thermally averaged observables.

    Assumes all relevant states are included. 
    Thus, no not check e.g. if the Boltzmann weight 
    of the last state is small.

    Parameters
    ----------
    energies : list(N)
        energies[i] is the energy of state i.
    observables : list(N,M)
        observables[i,j] is observable j of state i.
    T : float
        Temperature.
    tol : float
        Tolerance for smallest weight for the last energy.

    '''
    if len(energies) != np.shape(observable)[0]:
        raise ValueError("Passed array is not of the right shape")
    z = 0
    e_average = 0
    o_average = 0
    weights = np.zeros_like(energies)
    shift = np.min(energies)
    for j,(e,o) in enumerate(zip(energies,observable)):
        weight = np.exp(-(e-shift)/(k_B*T))
        z += weight
        e_average += weight*e
        o_average += weight*o
        weights[j] = weight
    e_average /= z
    o_average /= z
    weights /= z
    return o_average

def get_index_unique(x,xtol=0.001):
    """
    Return (first) indices of non degenerate values and corresponding degeneracy.

    """
    ind = []
    degen = []
    for i,e in enumerate(x):
        if i==0:
            ind.append(i)
            degen.append(1)
        elif e-x[i-1] > xtol:
            ind.append(i)
            degen.append(1)
        else:
            degen[-1] += 1
    return ind,degen

def get_spectrum(folder='.',xas_file='XASSpec.dat',
                 output_file='output.txt',T=300,
                 peak=None,loc=None,h=None,
                 peakorder='topdown',tol=4e-5):
    """
    Return the thermally averaged spectrum. 
    
    It can also be shifted in energy and scaled 
    in intensity by the parameters peak, loc and h.

    Checks if enough energies have been computed.
    The Boltzmann weight of the state with the highest 
    energy should be smaller than variable tol, otherwise
    a warning is printed.
    """
    xas = read_data_file(folder + '/' + xas_file)
    w = xas[:,0]
    energies = read_output_file(folder + '/' + output_file)[:,1]

    if tol < np.exp(-(energies[-1]-energies[0])/(k_B*T)):
        print('Warning: Perhaps too few eigenenergies considered.')
        print('E-E0 =')
        print(energies-energies[0])

    xasa = thermal_average(energies,np.transpose(xas[:,2::2]),T)
    if peak == None and loc == None and h == None:
        return w,xasa
    elif peak == None or loc == None or h == None:
        print('Either none or all of the parameters:'
              'peak, loc and h should be specified.')
        return w,xasa
    else:
        shift,scale = get_normalization(w,xasa,loc,h,
                                        peak=peak,
                                        peakorder=peakorder)
        return w + shift, scale*xasa

