#!/usr/bin/env python3

"""

RSPtT (Relativistic Spin Polarized toolkit Tools)
=================================================
   
This module contains functions which are useful to
process data generated from the RSPt software.
Material specific functions/variables should be avoided 
in this module.

"""

import numpy as np
import subprocess
import sys
from math import pi
from numba import jit

__author__ = "Johan Schott"
__email__ = "johan.schott@gmail.com"
__status__ = "Development"

# ----------------------------------------------------------
# Slater-Condon integrals
@jit
def get_Slater_F(r,f1,f2,k):
    r"""
    Return Slater integral :math:`F^{(k)}`.
    
    Calculate :math:`F^{(k)}` for radial functions `f1` and `f2`, 
    and the radial grid `r`.
    A just-in-time (JIT) compilation is used for speed-up reasons.
    
    Parameters
    ----------
    r : (M) array
        Radial grid.
    f1 : (M) array
        Radial function.
    f2 : (M) array
        Radial function.
    k : int
        Order of Slater integral to evaluate.

    Returns
    -------
    s : float
        Slater integral :math:`F^{(k)}`.
    
    Notes
    -----
    The Slater integrals are calculated according to [1]_:
    
    .. math:: F^{(k)} = \int_{0}^{\infty} \int_{0}^{\infty} dr dr' (r r' f_1(r) f_2(r'))^2 \frac{\mathrm{min}(r,r')^k}{\mathrm{max}(r,r')^{k+1}}

    .. [1] J. Luder, "Theory of L-edge spectroscopy of strongly correlated systems", Phys Rev. B 96, 245131 (2017). 

    Examples
    --------
    >>> import numpy as np
    >>> from math import pi
    >>> from py4rspt.rsptt import get_Slater_F
    >>> r = np.linspace(1e-6,1,300)
    >>> f1 = np.sin(2*pi*r)/np.sqrt(2)
    >>> f2 = np.sin(4*pi*r)/np.sqrt(2)
    >>> k = 0
    >>> print(get_Slater_F(r,f1,f2,k))
    0.00816015659426
    >>> k = 2
    >>> print(get_Slater_F(r,f1,f2,k))
    0.00503862918722

    """
    # Define integration trapezoidal integration weights 
    dr = np.zeros_like(r)
    dr[1:-1] = (r[2:]-r[:-2])/2.
    dr[0] = (r[1]-r[0])/2.
    dr[-1] = (r[-1]-r[-2])/2.
    n = len(r)
    s = 0
    for i in range(n):
        for j in range(n):
            s += (dr[i]*dr[j]*(r[i]*r[j]*f1[i]*f2[j])**2
                  *min(r[i],r[j])**k
                  /max(r[i],r[j])**(k+1))
    return s

@jit
def get_Slater_G(r,f1,f2,k):
    r"""
    Return Slater integral :math:`G^{(k)}`.
    
    Calculate :math:`G^{(k)}` for radial functions `f1` and `f2`, 
    and the radial grid `r`.
    A just-in-time (JIT) compilation is used for speed-up reasons.
    
    Parameters
    ----------
    r : (M) array
        Radial grid.
    f1 : (M) array
        Radial function.
    f2 : (M) array
        Radial function.
    k : int
        Order of Slater integral to evaluate.

    Returns
    -------
    s : float
        Slater integral :math:`G^{(k)}`.
    
    Notes
    -----
    The Slater integrals are calculated according to [1]_:
    
    .. math:: G^{(k)} = \int_{0}^{\infty} \int_{0}^{\infty} dr dr' (r r')^2 f_1(r) f_2(r) f_1(r') f_2(r') \frac{\mathrm{min}(r,r')^k}{\mathrm{max}(r,r')^{k+1}}
    
    .. [1] J. Luder, "Theory of L-edge spectroscopy of strongly correlated systems", Phys Rev. B 96, 245131 (2017). 

    Examples
    --------
    >>> import numpy as np
    >>> from math import pi
    >>> from py4rspt.rsptt import get_Slater_G
    >>> r = np.linspace(1e-6,1,300)
    >>> f1 = np.sin(2*pi*r)/np.sqrt(2)
    >>> f2 = np.sin(4*pi*r)/np.sqrt(2)
    >>> k = 1
    >>> print(get_Slater_G(r,f1,f2,k))
    0.000834095400633
    >>> k = 3
    >>> print(get_Slater_G(r,f1,f2,k))
    0.00121035494022

    """
    # Define integration trapezoidal integration weights 
    dr = np.zeros_like(r)
    dr[1:-1] = (r[2:]-r[:-2])/2.
    dr[0] = (r[1]-r[0])/2.
    dr[-1] = (r[-1]-r[-2])/2.
    n = len(r)
    s = 0
    for i in range(n):
        for j in range(n):
            s += (dr[i]*dr[j]*(r[i]*r[j])**2
                  *f1[i]*f2[i]*f1[j]*f2[j]
                  *min(r[i],r[j])**k
                  /max(r[i],r[j])**(k+1))
    return s

# ----------------------------------------------------------
# Functions for extracting data from RSPt generated files 
# or functions for nicely formatted printouts

def print_matrix(x,space=7,ndecimals=3,fmt='f',cutoff=True):
    """
    Return string representation of matrix for printing.

    Parameters
    ----------
    x : (M,N) array
        Matrix to convert to a string.
    space : int
        Space for each number.
    ndecimals : int
        Number of decimals.
    fmt : {'f', 'E'}
        Print format keyword.
    cutoff : boolean
        If True, small numbers are presented as 0.

    Returns
    -------
    s : str
        String representation of matrix.

    Examples
    --------
    >>> from py4rspt.rsptt import print_matrix
    >>> from numpy.random import rand
    >>> x = rand(5,4)
    >>> print(x)
    [[ 0.84266211  0.51373679  0.62017691  0.14055559]
    [ 0.63183783  0.06084673  0.05167614  0.16491208]
    [ 0.55515508  0.47868486  0.79075186  0.4892547 ]
    [ 0.40485259  0.65460802  0.62777336  0.71200114]
    [ 0.54512609  0.18695706  0.6019384   0.85743096]]
    >>> print(print_matrix(x))
      0.843  0.514  0.620  0.141
      0.632  0.061  0.052  0.165
      0.555  0.479  0.791  0.489
      0.405  0.655  0.628  0.712
      0.545  0.187  0.602  0.857
    
    """
    fmt_f = '{:' + str(space) + '.' + str(ndecimals) + 'f}'
    fmt_e = '{:' + str(space) + '.' + str(ndecimals) + 'E}'
    fmt_int = '{:' + str(space) + 'd}'
    
    if fmt == 'f':
        fmt_s = fmt_f
    elif fmt == 'E':
        fmt_s = fmt_e
        
    if cutoff:    
        s = []
        for row in x:
            rowl = []
            for item in row:
                if np.abs(item) < 0.5*10**(-ndecimals):
                    rowl.append(fmt_int.format(0))
                else:
                    rowl.append(fmt_s.format(item))
            s.append(''.join(rowl))
        s = '\n'.join(s)
    else:
        s = '\n'.join([''.join([fmt_s.format(item) for item in row]) 
              for row in x])
            
    return s

def parse_matrices(out_file='out',
                   search_phrase='Local hamiltonian'):    
    '''
    Return matrices and corresponding labels.
    
    Parameters
    ----------
    out_file : str
        File to read.
    search_phrase : str
        Search phrase for matrix.
    
    Returns
    -------
    hs : list
        List of matrices.
    labels : list
        List of labels.

    '''
    with open(out_file, 'r') as f:
        data = f.read()
    lines = data.splitlines()
    h_ids = []
    for i,line in enumerate(lines):
        if search_phrase in line:
            h_ids.append(i)
    # Store matrices
    hs = []
    # Store labels
    labels = []
    for h_id in h_ids:
        labels.append(lines[h_id].split()[1])
        # Real and imaginary part of matrix
        hr = []
        hi = []
        r_empty = False
        r_id = h_id+2
        imag = 1
        # Loop until get empty line
        while r_empty is False:
            if len(lines[r_id].split()) == 0:
                r_empty = True
            elif lines[r_id].split()[0][:4] == 'Imag':
                imag = 1j
                r_id += 1
            else:
                if imag == 1:                
                    hr.append(
                        [float(c) for c in lines[r_id].split()])
                else:
                    hi.append(
                        [float(c) for c in lines[r_id].split()])
                r_id += 1
        hr = np.array(hr)
        hi = np.array(hi)
        h = hr+hi*1j
        hs.append(h)
    return hs,labels
    
def represents_int(s):
    '''
    Return boolean about whether it is possible to 
    convert input parameter to an int.

    '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def parse_core_energies(path):
    '''
    Return all the core energies stored in RSPt out file.
    
    path - path to RSPt out file to parse.
    '''
    with open(path, 'r') as f:
        data = f.readlines()
    its = [] # indices for the different types
    ts = []  # types 
    # For each type, find first row about core energies
    for i,row in enumerate(data):
        if 'type:' in row:
            its.append(i)
            t = int(row.split()[1])
            ts.append(t)
    es = {} # energies for the different types
    # Loop over all types
    for it,t in zip(its,ts):
        # First row containing energies
        i = it+4
        es[t] = []
        while represents_int(data[i].split()[0]):
            es[t].append(float(data[i].split()[2]))
            i += 1
        es[t] = np.array(es[t])
    return es


# ----------------------------------------------------------
# Helpful function when generating a rotated basis 
# for the localized orbitals in RSPt  

def write_proj_file(x,spinpol=False,filename='proj-LABEL-IrrXXXX.inp', 
                    tol=1e-11):
    ''' 
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

    '''
    n = len(x[:,0])
    m = len(x[0,:])   
    if n != m:
        sys.exit('Square matrix expected')
    counter = 0
    for j in range(m):
        for i in range(n):
            if np.abs(x[i,j]) > tol:
                counter += 1
    with open(filename, 'w') as f:      
        if spinpol:
            f.write('{:d} {:d} {:d}'.format(n,n,counter))
        else:
            f.write('{:d} {:d} {:d}'.format(n,2*n,counter))
        f.write(' ! presize, dmftsize, #lines, each row has'
                ' (i,j,Re v[i,j],Im v[i,j]).')
        f.write('Eigenvectors on columns. \n')
        for j in range(m):
            for i in range(n):
                if np.abs(x[i,j]) > tol:
                    f.write(
                        '{:d} {:d} {:20.15f} {:20.15f} \n'.format(
                            i+1,j+1,x[i,j].real,x[i,j].imag))


# ----------------------------------------------------------
# Functions related to crystal-field splitting of localized orbitals

def integrate(x,y,xmin=None,xmax=None):
    '''
    Return numerical integration value.

    '''
    if (xmin is not None) or (xmax is not None):
        if xmin is None: 
            mask = x < xmax 
        elif xmax is None: 
            mask = xmin < x 
        else: 
            mask = (xmin < x) & (x < xmax) 
        x = x[mask]
        y = y[mask]
    return np.trapz(y,x)

def averageE(w,PDOS,wmin=None,wmax=None):
    '''
    Return the average energy within a energy window.
    
    Parameters
    ----------
    w : (N) array
        Energy vector.
    PDOS - (N) array
        Projected density of states vector.
    wmin : float
        Lower limit of energy window.
    wmax : float
        Upper limit of energy window.
    '''
    if (wmin is None) and (wmax is None):
        return integrate(w,PDOS*w)/integrate(w,PDOS)
    elif wmin is None: 
        return (integrate(w,PDOS*w,wmax=wmax)/
                integrate(w,PDOS,wmax=wmax))
    elif wmax is None: 
        return (integrate(w,PDOS*w,wmin=wmin)
                /integrate(w,PDOS,wmin=wmin))
    else:
        return (integrate(w,PDOS*w,wmin,wmax)
                /integrate(w,PDOS,wmin,wmax))

def get_CF_mean_energies(w,pdos_eg,pdos_t2g,wmin,wmax):
    '''
    Return the crystal-field splitting in cubic environment. 
    
    Symmetries can be octahedral (O_h) or tetragonal (T_d). 

    '''
    ega = averageE(w,pdos_eg,wmin,wmax)
    t2ga = averageE(w,pdos_t2g,wmin,wmax)
    return ega-t2ga,ega,t2ga

def get_D4h_splitting(w,pdos_x2y2,pdos_z2,pdos_xy,pdos_xz,
                      wmin,wmax):
    '''
    Return the crystal-field splitting in D_4h symmetry.
    
    Quadradic planar systems or tetragonally distored 
    octahedrons have this symmetry.

    '''
    x2y2 = averageE(w,pdos_x2y2,wmin,wmax)
    z2 = averageE(w,pdos_z2,wmin,wmax)
    xy = averageE(w,pdos_xy,wmin,wmax)
    xz = averageE(w,pdos_xz,wmin,wmax)
    e_mean = (x2y2 + z2 + xy + 2*xz)/5.
    deltao = 1/2.*x2y2 + 1/2.*z2 - 1/3.*xy - 2/3.*xz
    delta1 = x2y2 - z2
    delta2 = xy - xz
    return e_mean,deltao,delta1,delta2

def get_delta_o(e):
    r"""
    Return CF parameter: :math:`\delta_o = e_{e_g} - e_{t_{2g}}`. 
    
    Parameters
    ----------
    e : (N) array
        On-site energies.

    """
    if len(e) == 2:
        # Assumed order: eg and t2g
        eg = np.array(e[0])
        t2g = np.array(e[1])
        deltao = eg-t2g
    elif len(e) == 3:
        # Assumed order: eg1, eg2 and t2g 
        eg1 = np.array(e[0])
        eg2 = np.array(e[1])
        t2g = np.array(e[2])
        deltao = (eg1+eg2)/2. - t2g
    elif len(e) == 4:
        # Assumed order: z2, x2y2, xz=yz and xy
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        xz = np.array(e[2])
        xy = np.array(e[3])
        deltao = (z2+x2y2)/2. - (2*xz+xy)/3.
    return deltao 

def get_delta1(e):
    r"""
    Return CF parameter: :math:`\delta_1 = e_{x^2-y^2} - e_{z^2}`. 
    
    Works in O_h and D_4h symmetry.
    
    Parameters
    ----------
    e : (N) array
        On-site energies.
    
    """
    if len(e) == 2:
        # Assumed order: eg and t2g 
        eg = np.array(e[0])
        t2g = np.array(e[1])
        delta1 = np.zeros_like(eg)    
    elif len(e) == 3: 
        # Assumed order: z2, x2y2 and t2g 
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        delta1 = x2y2-z2
    elif len(e) == 4: 
        # Assumed order: z2, x2y2 and xz=yz and xy 
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        delta1 = x2y2-z2
    return delta1 

def get_delta2(e):
    r"""
    Return CF parameter: :math:`\delta_2 = e_{xy} - e_{xz}`. 
    
    Works in O_h and D_4h symmetry.

    Parameters
    ----------
    e : (N) array
        On-site energies.
    
    """
    if len(e) == 2:
        # assumed order: eg and t2g 
        eg = np.array(e[0])
        t2g = np.array(e[1])
        delta2 = np.zeros_like(eg)    
    elif len(e) == 3: 
        # assumed order: z2, x2y2 and t2g 
        z2 = np.array(e[0])
        delta2 = np.zeros_like(z2)
    elif len(e) == 4: 
        # assumed order: z2, x2y2 and xz=yz and xy 
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        xz = np.array(e[2])
        xy = np.array(e[3])
        delta2 = xy-xz
    return delta2 

def get_3d_energy(e):
    r"""
    Return CF parameter: average 3d on-site energy.
    
    Works in O_h and D_4h symmetry.

    Parameters
    ----------
    e : (N) array
        On-site energies.
    
    """
    if len(e) == 2:
        # Assumed order: eg and t2g
        eg = np.array(e[0])
        t2g = np.array(e[1])
        em = 1/5.*(2*eg+3*t2g)
    elif len(e) == 3:
        # Assumed order: eg1, eg2 and t2g
        eg1 = np.array(e[0])
        eg2 = np.array(e[1])
        t2g = np.array(e[2])
        em = 1/5.*(1*eg1+1*eg2+3*t2g)
    elif len(e) == 4:
        # Assumed order: z2, x2y2, xz=yz and xy
        z2 = np.array(e[0])
        x2y2 = np.array(e[1])
        xz = np.array(e[2])
        xy = np.array(e[3])
        em = 1/5.*(1*z2+1*x2y2+2*xz+1*xy)
    return em


# ----------------------------------------------------------
# Functions related to hybridization function for localized orbitals 

def get_v_simple(w,hyb,eb,width=0.5):
    '''
    Return hopping parameters, in a simple fashion.
    
    Integrate hybridization weight within a fixed energy
    window around peak.
    This weight is given to the bath state.
    
    The energy window, determined by width, 
    should be big enough to capture the main part
    of the hybridization in that area, but small enough so 
    no weight is contributing to different bath states.
    '''
    # Hopping strength parameters
    vb = []
    # Number of different correlated orbitals, 
    # e.g. e_g and t_2g
    norb = len(eb)
    # Loop over correlated obitals
    for i in range(norb): 
        vb.append([])
        # Loop over bath states
        for e in eb[i]: 
            mask = np.logical_and(e-width/2 < w, w < e+width/2)    
            vb[i].append(
                np.sqrt(np.trapz(-hyb[mask,i],w[mask])/np.pi))
    return np.array(vb)

def get_v_from_eb(w,hyb,eb,accept1=0.1,accept2=0.5,nbig=20):
    """
    Return hopping parameters.
    
    Integrate hybridization spectral weight within an energy 
    window around peak.
    This weight is given to the bath state.

    Parameters
    ----------
    w : (M) array
        Energy vector.
    hyb : (N,M) vector
        The imaginary part of hybridization function.
        Orbitals on seperate rows.
    eb : list
        List of list of bath energies for different 
        correlated orbitals.
    accept1 : float
        Parameter to determine the energy window.
    accept2 : float
        Parameter to determine the energy window.
    nbig : int
        Parameter to determine the energy window.

    Returns
    -------
    vb : list
        List of lists of hopping parameters for different 
        correlated orbitals.
    wborder : list
        Integration energy window borders for each 
        bath state.

    """
    # Make sure hyb is an numpy matrix instead of a list 
    # of lists
    if type(hyb) is list:
        hyb = np.array(hyb)
    # Hopping strength parameters
    vb = []
    # Integration borders
    wborder = []
    # Number of different correlated orbitals, 
    # e.g. e_g and t_2g
    nc = len(eb)
    # Loop over correlated obitals
    for i in range(nc):
        vb.append([])
        wborder.append([])
        # Get energy window border indices
        kmins,kmaxs = get_border_index(w,-hyb[i,:],eb[i],
                                       accept1,accept2,nbig)
        # Loop over bath states
        for j,e in enumerate(eb[i]):
            kmin = kmins[j]
            kmax = kmaxs[j]
            wborder[i].append([w[kmin],w[kmax]])
            vb[i].append(np.sqrt(
                np.trapz(-hyb[i,kmin:kmax],w[kmin:kmax])/np.pi))
    return np.array(vb),np.array(wborder)

def get_border_index(x,y,eb,accept1,accept2,nbig):
    r"""
    Return the lower/left and upper/right (integration) 
    limit indices.
    
    First, the left and the right limits are determined 
    independently of each other.
    For both limits, three criteria is used to determine 
    the limit position.
    1) Look for intensity drop to `accept1` times the value 
    in `y` at the bath energy.
    2) Look for intensity minimum in `y` between neighbouring 
    bath energy, which is lower than `accept2` times the 
    value of `y` at the bath energy.
    3) Pick limit halfway to the other bath energy.

    - Criterion one is initially tried.
    - If it is successful, it returns the limit position.
    - If not successful, the second criterion is tried.
    - If it is successful, it returns the limit position.
    - If not successful, the third criterion is used as a 
      final resort.

    To avoid energy windows to overlap,
    the borders of the energy windows are checked
    and in the case of an overlap,
    instead the mean of the overlapping border energies are 
    used as border.

    If there are more than a big number of bath states,
    the edge bath states are specially treated.
    This is to avoid the integration windows for the edge bath
    states to become unreasonably big.
    Without special treatment, this problem arises if the
    hybridization intensity at the edge bath state is
    small (but not very small).
    Then the criteria p1 or p2 will be satisfied,
    but at an energy really far away from the bath location.
    Instead, with the special treatment, the border for
    the edge bath state is determined by the distance to
    the nearest bath state.

    Parameters
    ----------
    x : (N) array
        Representing the energy mesh.
    y : (N) array
        Corresponding values to the energies in x.
    eb : (M) array
        Array of bath energies
    accept1 : float
        A procentage parameter used for criterion 1.
        This value times the value of y at the bath 
        location sets accepted minimum value of y.
    accept2 : float
        A procentage parameter used for criterion 2.
        This value times the value of y at the bath 
        location sets accepted minimum value of y.
    nbig : int
        If the number of bath states exceeds this 
        number, the edge bath states are specially 
        treated.
    
    Returns
    -------
    kmins_new : list
        List of left limit indices.
    kmaxs_new : list
        List of left limit indices.

    """
    # Index vectors to be returned
    kmins = []
    kmaxs = []
    # Loop over the bath energies
    for e_index,e in enumerate(eb):
        # The index corresponding to the bath energy
        kc = np.argmin(np.abs(x-e))
        # Accepted peak intensity, according to criterion 
        # 1 and 2, respectively
        p1,p2 = accept1*y[kc],accept2*y[kc]
        other_peaks = np.delete(np.array(eb),e_index)
        # Find left border
        if np.any(other_peaks < e):
            left_peak_index = np.argmin(np.abs(
                e - other_peaks[other_peaks < e]))
            k_left = np.argmin(np.abs(
                other_peaks[other_peaks < e][left_peak_index] - x))
        else:
            k_left = 0
        # Check if bath state is an edge bath state
        if k_left == 0 and len(eb)>nbig:
            # Pick point at distance equal to the
            # Distance to the bath energy to the right
            de = np.min(other_peaks-e)
            kmin = np.argmin(np.abs(x-(e-de)))
        else:
            # Look for intensity lower than p1
            for k in np.arange(kc,k_left,-1):
                if y[k] < p1:
                    kmin = k
                    break
            else:
                # Another bath energy was reached.
                # Therefore, look for intensity minimum 
                # between them, which should be lower than p2
                for k in np.arange(kc,k_left+2,-1):
                    if (y[k-1]<y[k] and y[k-1]<y[k-2]) and y[k]<p2:
                        kmin = k-1
                        break
                else:
                    # There is no intensity minimum.
                    if k_left == 0 and len(other_peaks)>0:
                        # Pick point at distance equal to the
                        # distance to the bath energy to the 
                        # right
                        de = np.min(other_peaks-e)
                        kmin = np.argmin(np.abs(x-(e-de)))
                    else:
                        # Pick point halfway between them.
                        kmin = np.argmin(np.abs(
                            x-(x[k_left]+x[kc])/2))

        # find right border
        if np.any(other_peaks > e):
            right_peak_index = np.argmin(np.abs(
                e - other_peaks[other_peaks > e]))
            k_right = np.argmin(np.abs(
                other_peaks[other_peaks > e][right_peak_index] - x))
        else:
            k_right = len(x)-1
        # Check if bath state is an edge bath state
        if k_right == len(x)-1 and len(eb)>nbig:
            # Pick point at distance equal to the
            # distance to the bath energy to the left
            de = np.min(e-other_peaks)
            kmax = np.argmin(np.abs(x-(e+de)))
        else:
            # look for intensity lower than p1
            for k in np.arange(kc,k_right):
                if y[k] < p1:
                    kmax = k
                    break
            else:
                # Another bath energy was reached.
                # Therefore, look for intensity minimum 
                # between them, which should be lower than p2
                for k in np.arange(kc,k_right-2):
                    if (y[k+1]<y[k] and y[k+1]<y[k+2]) and y[k]<p2:
                        kmax = k+1
                        break
                else:
                    # There is no intensity minimum.
                    if k_right == len(x)-1 and len(other_peaks)>0:
                        # Pick point at distance equal to the
                        # distance to the bath energy to the 
                        # left
                        de = np.min(e-other_peaks)
                        kmax = np.argmin(np.abs(x-(e+de)))
                    else:
                        # Pick point halfway between them.
                        kmax = np.argmin(np.abs(
                            x-(x[kc]+x[k_right])/2))
        kmins.append(kmin)
        kmaxs.append(kmax)
    # copy the lists
    kmins_new = list(kmins)
    kmaxs_new = list(kmaxs)
    # check for overlaps, by looping over the bath energies
    for e_index,(e,kmin,kmax) in enumerate(zip(eb,kmins,kmaxs)):
        # loop over the other bath energies
        for i in range(e_index) + range(e_index+1,len(eb)):
            # check for overlap to the left
            if eb[i] < e and kmin < kmaxs[i]:
                kmins_new[e_index] = np.argmin(np.abs(
                    x-(x[kmin]+x[kmaxs[i]])/2.))
            # check for overlap to the right
            if eb[i] > e and kmax > kmins[i]:
                kmaxs_new[e_index] = np.argmin(np.abs(
                    x-(x[kmax]+x[kmins[i]])/2.))
    return kmins_new,kmaxs_new

def hyb_d(z,eb,vb):
    '''
    return the hybridization function at points z.
    
    Several independent impurity orbitals,
    e.g. e_g and t_2g, possible.

    Parameters
    ----------
    z : (M) array
        Vector containing points in the complex plane.
    eb : (N,K) array
        Bath energies.
    vb : (N,K) array
        Hopping parameters.

    Returns
    -------
    d : {(M) ndarray, (N,M) ndarray}
        Hybridization function.
    '''
    eb = np.atleast_2d(eb)
    vb = np.atleast_2d(vb)
    # Number of different impurity orbitals
    (norb,nb) = np.shape(eb)
    # Hybridization function
    d = np.zeros((norb,len(z)),dtype=np.complex) 
    # Loop over correlated obitals
    for i in range(norb): 
        # Loop over bath states
        for e,v in zip(eb[i],vb[i]): 
            d[i,:] += np.abs(v)**2/(z-e) 
    if norb == 1:
        return d[0,:]
    else:
        return d


# ----------------------------------------------------------
# Double countings

def dc_MLFT(n3d_i,c,Fdd,n2p_i=None,Fpd=None,Gpd=None):
    r"""
    Return double counting (DC) in multiplet ligand field theory.
    
    Parameters
    ----------
    n3d_i : int
        Nominal (integer) 3d occupation.
    c : float
        Many-body correction to the charge transfer energy.
    n2p_i : int
        Nominal (integer) 2p occupation.
    Fdd : list
        Slater integrals {F_{dd}^k}, k \in [0,1,2,3,4] 
    Fpd : list
        Slater integrals {F_{pd}^k}, k \in [0,1,2] 
    Gpd : list
        Slater integrals {G_{pd}^k}, k \in [0,1,2,3] 
    
    Notes
    -----
    The `c` parameter is related to the charge-transfer 
    energy :math:`\Delta_{CT}` by:

    .. math:: \Delta_{CT} = (e_d-e_b) + c.

    """
    if not int(n3d_i) == n3d_i:
        raise ValueError('3d occupation should be an integer')
    if n2p_i != None and int(n2p_i) != n2p_i:
        raise ValueError('2p occupation should be an integer')

    # Average repulsion energy defines Udd and Upd
    Udd = Fdd[0] - 14.0/441*(Fdd[2] + Fdd[4])
    if n2p_i==None and Fpd==None and Gpd==None:
        return Udd*n3d_i - c
    if n2p_i==6 and Fpd!=None and Gpd!=None:
        Upd = Fpd[0] - (1/15.)*Gpd[1] - (3/70.)*Gpd[3]
        return [Udd*n3d_i+Upd*n2p_i-c,Upd*(n3d_i+1)-c]
    else:
        raise ValueError('double counting input wrong.')

def dc_FLL(n,F0,F2,F4):
    r"""
    Return double counting in the fully localized limit.
    
    Parameters
    ----------
    n : float
        Occupation of 3d orbitals.
    F0 : float
        Slater integral :math:`F_{dd}^{(0)}`.
    F2 : float
        Slater integral :math:`F_{dd}^{(2)}`.
    F4 : float
        Slater integral :math:`F_{dd}^{(4)}`.

    """
    J = 1/14.*(F2+F4)
    return F0*(n-1/2.)-J/2.*(n-1) 


# ----------------------------------------------------------
# Functions related to impurity PDOS
# For example there is a function calculating the PDOS 
# using the single-particle Hamiltonian $H_0$. 
# And another using the hybridization function.

def lorentzian(w,wc,eim):
    '''
    Return lorentzian with center at wc and with width 
    given by eim.

    '''
    return 1/pi*eim/((w-wc)**2+eim**2)

def mu(path='out'):
    '''
    Return the chemical potential.
    
    Parsed from file.
    If possible, the function greps for the 
    'green_mu ' keyword, otherwise it will 
    grep for the 'fermi energy' keyword. 

    Parameters
    ----------
    path : str
        Filename of file to parse.
    
    '''
    try:
        mu = subprocess.check_output("grep 'green_mu ' "+path,
                                     shell=True)
        mu = float(mu.split()[2])
    except subprocess.CalledProcessError:
        mu = subprocess.check_output("grep 'fermi energy' "+path,
                                     shell=True)
        mu = float(mu.split()[3])
    return mu

@jit
def pdos(w,eim,e,hyb,sig=0):
    r"""
    Return impurity projected density of states (PDOS).
    
    Parameters
    ----------
    w : (M) array
        Energy mesh :math:`\omega`.
    eim : float
        Distance :math:`\delta` above real-energy axis.
    e : {(N) array, (N,N) array}
        If (N) array: diagonal on-site energies
        If (N,N) array: full on-site matrix
    hyb : {(N,M) array, (N,N,M) array}
        Hybridization function :math:`\Delta(\omega+i\delta)`.
        If (N,M) array: diagonal hybridization function.
        If (N,N,M) array: full hybridization function.
    sig : {(N,M) array, (N,N,M) array}
        Self-energy :math:`\Sigma(\omega+i\delta)`.
        If (N,M) array: If equal dimensions, static self-energy,
        otherwise treated as diagonal but dynamical. 
        If (N,N,M) array: full and dynamical self-energy. 

    Returns
    -------
    pdos : (N,N,M) ndarray
        Calculated PDOS.    

    .. math:: PDOS_{a,b}(\omega) = (((\omega + i \delta)\delta_{i,j}-e_{i,j}-\Delta_{i,j}(\omega+i\delta)-\Sigma_{i,j}(\omega+i\delta)  )^{-1})_{a,b} 
    
    """
    e = np.array(e)
    # Number of correlated orbitals
    n = np.shape(e)[0]
    nw = len(w)
    if isinstance(sig,int) and sig == 0:
        sig = np.zeros((n,nw))
    # If everything is diagonal
    diag = (e.ndim == 1 and hyb.ndim == 2 
            and np.shape(sig)[0] != np.shape(sig)[1])
    if diag:
        g = np.zeros((n,nw),dtype=np.complex)
        for i in range(n):
            g[i,:] = 1./(w[:]+1j*eim-e[i]-hyb[i,:]-sig[i,:])
    else:
        # Transform everything to off-diagonal 
        # Make on-site energy 2d
        e = e if e.ndim == 2 else np.diag(e) 
        # Make hybridization 3d
        if hyb.ndim == 2:
            tmp = np.zeros((n,n,nw),dtype=np.complex)
            for i in range(n):
                tmp[i,i,:] = hyb[i,:]
            hyb = tmp
        # Make self-energy 3d
        if sig.ndim == 2:
            tmp = np.zeros((n,n,nw),dtype=np.complex)
            if np.shape(sig)[0] == np.shape(sig)[1]:
                for i in range(nw):
                    tmp[:,:,i] = sig
            else:
                for i in range(n):
                    tmp[i,i,:] = sig[i,:]
            sig = tmp
        assert nw == np.shape(hyb)[2] and nw == np.shape(sig)[2]
        g = np.zeros((n,n,nw),dtype=np.complex)
        for i,x in enumerate(w):
            g[:,:,i] = np.linalg.inv((x+1j*eim)*np.eye(n)
                                     -e[:,:]
                                     -hyb[:,:,i]
                                     -sig[:,:,i])
    pdos = -1/pi*np.imag(g)
    return pdos

def pdos_d0_1(w,eim,hd0):
    """
    Return non-interacting impurity PDOS for 
    the first orbital of each impurity type.

    Parameters
    ----------
    w : (N) array
        Energy mesh.
    eim : float
        Distance above real-energy axis.   
    hd0 : {(N,N) array, (M,N,N) array} 
        Single-particle Hamiltonian, 
        can contain several impurity types
        (e.g. eg and t2g).

    """
    if len(np.shape(hd0)) == 2:
        hd0 = [hd0]
    # number of types
    n = np.shape(hd0)[0]
    pdos = np.zeros((n,len(w)))
    # loop over impurity types
    for t in range(n):
        eig,v = np.linalg.eigh(hd0[t])   
        for e,weight in zip(eig,np.abs(v[0,:])**2):
            pdos[t,:] += weight*lorentzian(w,e,eim)        
    if n == 1:
        return pdos[0]
    else:
        return pdos

def eig_and_weight1(h0):
    '''
    Return eigenvalues and weights for the first orbital,
    for each type.
    
    Parameters
    ----------
    h0 : (..., M, M) array
        Single-particle Hamiltonian, 
        can contain several independent types 
        (e.g. eg and t2g) 
    
    Returns
    -------
    eig : (...,M) ndarray     
        The eigenvalues. 
        If one type, (M,) ndarray
        If many types, (N,M) ndarray
    '''
    h0 = np.atleast_3d(h0)
    # number of types
    (nt,nh) = np.shape(h0)[:-1]
    eig = np.zeros((nt,nh))
    weight = np.zeros((nt,nh))
    # loop over types
    for t in range(nt):
        e,v = np.linalg.eigh(h0[t])   
        eig[t,:] = e
        weight[t,:] = np.abs(v[0,:])**2        
    if nt == 1:
        return eig[0,:],weight[0,:]
    else:
        return eig,weight

def h_d0(e,eb=None,v=None):
    '''
    Return single-particle Hamiltonian.
    
    Many independent impurity types,
    (e.g. eg and t2g), are possible.
    
    Parameters
    ----------
    e : (N) array
        Impurity energy level.
    eb : (N,M) array
        Bath energies.
    v : (N,M) array
        Hopping strengths.

    '''
    # Transform to eventually one bigger dimension
    # in order to conveniently treat one or many 
    # impurity orbitals 
    e = np.atleast_1d(e)
    if (eb is None and v is None) or (len(eb)==0 and len(vb)==0):
        # no bath states
        eb = [[]]*len(e) 
        v = [[]]*len(e)
    else:
        eb = np.atleast_2d(eb)
        v = np.atleast_2d(v)
    # number of types and bath states
    (nt,nb) = np.shape(eb)
    ht = []
    # loop over the types
    for t in range(nt):
        # create the sub-block Hamiltonian
        h = np.atleast_2d(np.zeros((1+nb,1+nb),dtype=np.complex))
        h[0,0] = e[t]
        for i,e_bath in enumerate(eb[t]):
            h[1+i,1+i] = e_bath
        for i,vb in enumerate(v[t]):
            h[1+i,0] = vb
            # it could actually be complex conjugate 
            # here but then H needs to be defined as complex
            h[0,1+i] = np.conj(vb)
        ht.append(h)     
    if nt == 1:
        return ht[0]
    else:
        return ht

