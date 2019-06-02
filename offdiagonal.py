#!/usr/bin/env python3

"""

offdiagonal
===========

This module contains functions to treat
off-diagonal hybridization functions.

"""

import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import minimize

from .energies import cog

def plot_diagonal_and_offdiagonal(w, hyb_diagonal, hyb, xlim):
    """
    Plot diagonal and offdiagonal hybridization functions separately.
    """
    # Number of considered impurity orbitals
    n_imp = np.shape(hyb_diagonal)[0]

    # Plot mask
    mask = np.logical_and(xlim[0] < w, w < xlim[1])
    # Diagonal functions
    # Real part
    plt.figure()
    for i in range(n_imp):
        plt.plot(w[mask], hyb_diagonal[i,mask].real, label=str(i))
    plt.legend()
    plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.title('Diagonal functions, real part')
    plt.show()
    # -Imag part
    plt.figure()
    for i in range(n_imp):
        plt.plot(w[mask], -hyb_diagonal[i,mask].imag, label=str(i))
    plt.legend()
    plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.title('Diagonal functions, -imag part')
    plt.show()

    # Off diagonal functions
    # Real part
    plt.figure()
    for i in range(n_imp):
        for j in list(range(i)) + list(range(i+1, n_imp)):
            plt.plot(w[mask], hyb[i,j,mask].real)
    plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.title('Off diagonal functions, real part')
    plt.show()
    # -Imag part
    plt.figure()
    for i in range(n_imp):
        for j in list(range(i)) + list(range(i+1, n_imp)):
            plt.plot(w[mask], -hyb[i,j,mask].imag)
    plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.title('Off diagonal functions, -imag part')
    plt.show()


def plot_all_orbitals(w, hyb_orig, hyb_model=None, xlim=None):
    """
    Plot functions for all orbitals, for both hyb and hyb_model.
    """
    assert np.shape(hyb_orig)[0] == np.shape(hyb_orig)[1]
    if hyb_model is not None:
        assert np.shape(hyb_orig) == np.shape(hyb_model)
    if xlim is None:
        mask = np.ones_like(w, dtype=np.bool)
    else:
        # Mask for plotting
        mask = np.logical_and(xlim[0] < w, w < xlim[1])
    # Number of rows and columns in the figure
    n = np.shape(hyb_orig)[0]
    if n > 1:
        # All real functions
        fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
        for i in range(n):
            for j in range(n):
                axes[i,j].plot(w[mask], hyb_orig[i,j,mask].real, label='original')
                if hyb_model is not None:
                    axes[i,j].plot(w[mask], hyb_model[i,j,mask].real, label='model')
                #axes[i,j].grid()
        if xlim is not None:
            plt.xlim(xlim)
        #plt.ylim(ylim)
        axes[0,n//2].set_title('Real part')
        #plt.tight_layout()
        plt.subplots_adjust(top=0.95,right=0.98, wspace=0.03, hspace=0.03)
        plt.show()

        # All -imag functions
        fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
        for i in range(n):
            for j in range(n):
                axes[i,j].plot(w[mask], -hyb_orig[i,j,mask].imag, label='original')
                if hyb_model is not None:
                    axes[i,j].plot(w[mask], -hyb_model[i,j,mask].imag, label='model')
                #axes[i,j].grid()
        if xlim is not None:
            plt.xlim(xlim)
        #plt.ylim(ylim)
        axes[0,-1].legend()
        axes[0,n//2].set_title('- Imag part')
        #plt.tight_layout()
        plt.subplots_adjust(top=0.95,right=0.98, wspace=0.03, hspace=0.03)
        plt.show()
    elif n == 1:
        # All real functions
        fig = plt.figure()
        for i in range(n):
            for j in range(n):
                plt.plot(w[mask], hyb_orig[i,j,mask].real, label='original')
                if hyb_model is not None:
                    plt.plot(w[mask], hyb_model[i,j,mask].real, label='model')
                #plt.grid()
        if xlim is not None:
            plt.xlim(xlim)
        #plt.ylim(ylim)
        plt.title('Real part')
        plt.tight_layout()
        #plt.subplots_adjust(top=0.95,right=0.98, wspace=0.03, hspace=0.03)
        plt.show()

        # All -imag functions
        fig = plt.figure()
        for i in range(n):
            for j in range(n):
                plt.plot(w[mask], -hyb_orig[i,j,mask].imag, label='original')
                if hyb_model is not None:
                    plt.plot(w[mask], -hyb_model[i,j,mask].imag, label='model')
                #plt.grid()
        if xlim is not None:
            plt.xlim(xlim)
        #plt.ylim(ylim)
        plt.legend()
        plt.title('- Imag part')
        plt.tight_layout()
        #plt.subplots_adjust(top=0.95,right=0.98, wspace=0.03, hspace=0.03)
        plt.show()
    else:
        sys.exit('Positive number of impurity orbitals required.')


def get_eb_v_for_one_block(w, eim, hyb, block, wsparse, wborders,
                           n_bath_sets_foreach_window, xlim=None,
                           verbose_fig=False, gamma=0.):
    """
    Return bath energies and hybridization hopping parameters
    for a specific block.

    """
    # Select energies in real axis mesh.
    w_select = w[::wsparse]
    assert w_select[1] - w_select[0] < 2*eim
    # For each block, fit a discretized hybridization function to
    # the original hybridization function.
    # Number of bath orbitals, for each energy window.
    n_bath_foreach_window = np.array(n_bath_sets_foreach_window)*len(block)
    # Select a subset of all impurity orbitals
    hyb_block = hyb[block[:, np.newaxis], block, ::wsparse]
    print('shape(hyb_block) = ', hyb_block.shape)
    print('Get bath energies...')
    ebs = get_ebs(w_select, hyb_block, wborders, n_bath_foreach_window)
    #print('shape(ebs) = ', ebs.shape)
    #print('ebs:')
    #print(ebs)
    print('Get hopping parameters...')
    mask_tmp = np.logical_and(np.min(wborders) < w_select,
                              w_select < np.max(wborders))
    n_data_points = len(block)**2*len(w_select[mask_tmp])
    print('Fit to approx {:d} data points.'.format(n_data_points))
    n_param = np.sum(n_bath_foreach_window)*len(block)*2
    print('Use {:d} real-valued parameters in the fit.'.format(n_param))
    vs, costs = get_vs(w_select+1j*eim, hyb_block, wborders, ebs, gamma=gamma)
    print('Cost function values (without regularization):')
    print(costs)
    #print('shape(vs) = ', vs.shape)
    #print('vs:')
    #print(vs)
    #print('Get merged bath energies...')
    eb = merge_ebs(ebs)
    #print(eb)
    #print('Get merged hopping parameters...')
    v = merge_vs(vs)
    #print(v)
    if verbose_fig:
        print('Get model hybridization functions...')
        hyb_model = get_hyb(w_select+1j*eim, eb, v)
        print('Plot model and original hybridization functions..')
        plot_all_orbitals(w_select, hyb_block, hyb_model, xlim)
        # Distribution of hopping parameters
        plt.figure()
        plt.hist(np.abs(v).flatten()/np.max(np.abs(v)),bins=30)
        plt.xlabel('|v|/max(|v|)')
        plt.show()
        # Absolute values of the hopping parameters
        plt.figure()
        plt.plot(sorted(np.abs(v).flatten())/np.max(np.abs(v)),'-o')
        plt.ylabel('|v|/max(|v|)')
        plt.show()
        print('{:d} elements in v.'.format(v.size))
        v_mean = np.mean(np.abs(v))
        v_median = np.median(np.abs(v))
        print('<v> = ', v_mean)
        print('v_median = ', v_median)
        r_cutoff = 0.02
        mask = np.abs(v) < r_cutoff*np.max(np.abs(v))
        print('{:d} elements in v are smaller than {:.3f}*v_max.'.format(
            v[mask].size, r_cutoff))
        #print('Absolut values of these elements:')
        #print(sorted(np.abs(v[mask])))
    return eb, v


def get_eb_v(w, eim, hyb, blocks, wsparse, wborders,
             n_bath_sets_foreach_block_and_window, xlim=None,
             verbose_fig=False, gamma=0.):
    """
    Return bath and hopping parameters by discretizing hybridization functions.
    """
    # Number of considered impurity orbitals
    n_imp = sum(len(block) for block in blocks)
    # Bath energies
    eb = []
    # Hopping parameters
    v = []
    # Loop over blocks
    for block_i, (block, n_bath_sets_foreach_window) in enumerate(zip(blocks, n_bath_sets_foreach_block_and_window)):
        print('\n ---------------------------- \n')
        print('Block {:d} treats impurity orbitals:'.format(block_i), block)
        # Calculate bath energies and hopping parameters for each block.
        eb_block, v_block = get_eb_v_for_one_block(w, eim, hyb, block, wsparse,
                                                   wborders,
                                                   n_bath_sets_foreach_window,
                                                   xlim, verbose_fig,
                                                   gamma=gamma)
        eb.append(eb_block)
        v_sparse = np.zeros((len(eb_block), n_imp), dtype=np.complex)
        v_sparse[:, block] = v_block
        v.append(v_sparse)
    eb = np.hstack(eb)
    v = np.vstack(v)
    # Sort bath states according to the energy windows.
    # For example, the bath states with energies in the range of
    # the first energy window is placed first in the sorted list.
    # This is important if have both occupied and unoccupied bath states,
    # since we then want the unoccupied bath states to be sorted after
    # the occupied bath states.
    eb, v = reshuffle(eb, v, wborders)
    return eb, v


def reshuffle(eb, v, wborders):
    """
    Sort bath states according to the energy windows.

    For example, the bath states with energies in the range of
    the first energy window is placed first in the sorted list.
    This is important if have both occupied and unoccupied bath states,
    since we then want the unoccupied bath states to be sorted after
    the occupied bath states.
    """
    eb_new = []
    v_new = []
    for i, wborder in enumerate(wborders):
        mask = np.logical_and(wborder[0] <= eb, eb < wborder[1])
        eb_new.append(eb[mask])
        v_new.append(v[mask,:])
    eb_new = np.hstack(eb_new)
    v_new = np.vstack(v_new)
    return eb_new, v_new


def get_eb(w, hyb, n_b):
    """
    Return bath energies.

    Parameters
    ----------
    w : array(M)
        Real part of energy mesh.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    n_b : int
        Number of bath orbitals per window.

    Returns
    -------
    eb : array(n_b)
        Bath energies.

    """
    n_w = len(w)
    n_imp = np.shape(hyb)[0]
    eb = np.zeros(n_b, dtype=np.float)
    # Selection of bath energies depends on how many
    # bath orbitals have compared to the number of
    #impurity orbitals.
    if n_b <= 0:
        sys.exit('Positive number of bath energies expected.')
    elif n_b == 1:
        # Bath energy at the center of gravity of the imaginary part
        # of the hybridization function trace.
        eb[:] = cog(w, -np.trace(hyb.imag))
    elif n_b % n_imp == 0:
        # Remove the False variable below to activate a special treatment
        # of the case n_b == n_imp.
        if n_b == n_imp and False:
            # Bath energies at the center of gravities of each
            # diagonal hybridization function (its imaginary part).
            for i in range(n_b):
                eb[i] = cog(w, -hyb[i,i,:].imag)
        else:
            # Uniformly distribute n_b // n_imp energies on the mesh.
            dw = (w[-1] - w[0])/(n_b/n_imp + 1)
            es = np.linspace(w[0]+dw, w[-1]-dw, n_b//n_imp)
            # Give each energy degeneracy n_imp.
            for i, e in enumerate(es):
                eb[n_imp*i:n_imp*(1+i)] = e
    else:
        # Uniformly distribute the bath energies on the mesh.
        dw = (w[-1] - w[0])/(n_b + 1)
        eb = np.linspace(w[0]+dw, w[-1]-dw, n_b)
    return eb


def get_ebs(w, hyb, wborders, n_b):
    """
    Return bath energies, for each energy window.

    Parameters
    ----------
    w : array(M)
        Real part of energy mesh.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    wborders : array(K, 2)
        Window borders.
    n_b : array(K)
        Number of bath orbitals for each window.

    Returns
    -------
    ebs : tuple(K)
        Bath energies, for each energy window.
        Each element contains the bath energies for that energy window,
        as an array. len(ebs[i]) == n_b[i]


    """
    n_w = len(w)
    n_imp = np.shape(hyb)[0]
    n_windows = np.shape(wborders)[0]
    #ebs = np.zeros((n_windows, n_b), dtype=np.float)
    ebs = []
    # Treat each energy window as seperate.
    for a, wborder in enumerate(wborders):
        mask = np.logical_and(wborder[0]<= w, w <= wborder[1])
        #ebs[a,:] = get_eb(w[mask], hyb[:,:,mask], n_b)
        ebs.append(get_eb(w[mask], hyb[:,:,mask], n_b[a]))
    return ebs


def get_hyb(z, eb, v):
    """
    Return the hybridization functions, as a rank 3 tensor.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh.
    eb : array(B)
        Bath energies.
    v : array(B, N)
        Hopping parameters.

    Returns
    -------
    hyb : array(N,N,M)
        Hybridization functions.

    """
    n_w = len(z)
    n_b, n_imp = np.shape(v)
    hyb = np.zeros((n_imp, n_imp, n_w), dtype=np.complex)
    # Loop over all bath energies
    for b, e in enumerate(eb):
        # Add contributions from each bath
        hyb += np.outer(v[b,:].conj(), v[b,:])[:,:,np.newaxis]*(1/(z-e))
    return hyb


def get_vs(z, hyb, wborders, ebs, gamma=0.):
    """
    Return optimized hopping parameters.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh, just above the real axis.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    wborders : array(K, 2)
        Window borders.
    ebs : tuple(K)
        Bath energies, for each energy window.
        Each element contains the bath energies for that energy window,
        as an array(B), where B is different for each element.
    gamma : float
        Regularization parameter

    Returns
    -------
    vs : tuple(K)
        Hopping parameters, for each energy window.
        Each element in an array(B, N), where B is different for each element.
    costs : array(K)
        Cost function values, without regularization, for each energy window.

    """
    n_w = len(z)
    n_imp = np.shape(hyb)[0]
    #n_windows, n_b = np.shape(ebs)
    n_windows = len(ebs)
    # Hopping parameters
    #vs = np.zeros((n_windows, n_b, n_imp), dtype=np.complex)
    vs = []
    # Cost function values
    costs = np.zeros(n_windows, dtype=np.float)
    # Treat each energy window as seperate.
    for a, wborder in enumerate(wborders):
        mask = np.logical_and(wborder[0]<= z.real, z.real <= wborder[1])
        #vs[a,:,:], costs[a] = get_v(z[mask], hyb[:,:,mask], ebs[a,:], gamma)
        v, costs[a] = get_v(z[mask], hyb[:,:,mask], ebs[a], gamma)
        vs.append(v)
    return vs, costs


def get_v(z, hyb, eb, gamma=0.):
    """
    Return optimized hopping parameters.

    Parameters
    ----------
    z : complex array(M)
        Energy mesh, just above the real axis.
    hyb : array(N,N,M)
        RSPt hybridization functions.
    eb : array(B)
        Bath energies.
    gamma : float
        Regularization parameter

    Returns
    -------
    v : array(B, N)
        Hopping parameters.

    """
    n_w = len(z)
    n_imp = np.shape(hyb)[0]
    n_b = len(eb)
    # Initialize hopping parameters.
    # Treat complex-valued parameters,
    # by doubling the number of parameters.
    p0 = np.random.randn(2*n_b*n_imp)

    # Define cost function as a function of a hopping parameter
    # vector.
    fun = lambda p: cost_function(p, eb, z, hyb, gamma, output='value')
    jac = lambda p: cost_function(p, eb, z, hyb, gamma, output='gradient')
    # Minimize cost function
    res = minimize(fun, p0, jac=jac)
    #res = minimize(fun, p0)
    # The solution
    p = res.x
    # Cost function value, with regularization.
    c = cost_function(p, eb, z, hyb, output='value')
    # Convert hopping parameters to physical shape.
    v = unroll(p, n_b, n_imp)
    return v, c


def unroll(p, n_b, n_imp):
    """
    Return hybridization parameters as a matrix.

    Parameters
    ----------
    p : real array(K)
        Hybridization parameters as a vector.
    n_b : int
        Number of bath orbitals.
    n_imp : int
        Number of impurity orbitals.

    Returns
    -------
    v : complex array(n_b, n_imp)
        Hybridization parameters as a matrix.

    """
    assert len(p) % 2 == 0
    # Number of complex-value parameters
    r = len(p)//2
    p_c = p[:r] + 1j*p[r:]
    v = p_c.reshape(n_b, n_imp)
    return v


def inroll(v):
    """
    Return hybridization parameters as a vector.

    Parameters
    ----------
    v : complex array(n_b, n_imp)
        Hybridization parameters as a matrix.

    Returns
    -------
    p : real array(K)
        Hybridization parameters as a vector.

    """
    p = np.hstack((v.real.flatten(),
                   v.imag.flatten()))
    return p


def cost_function(p, eb, z, hyb, gamma=0., only_imag_part=True,
                  output='value and gradient', regularization_mode='L1'):
    """
    Return cost function value.

    Since the imaginary part of the hybridization function,
    from one bath state, is more local in energy
    than the real part, the default is to only fit to the imaginary part.

    Parameters
    ----------
    p : real array(K)
        Hopping parameters.
    eb : array(B)
        Bath energies.
    z : complex array(M)
        Energy mesh, just above the real axis.
    hyb : complex array(N,N,M)
        RSPt hybridization functions.
    only_imag_part : bool
        If should consider only the imaginary part of the
        hybridization functions, or consider both real and
        imaginary part.
    gamma : float
        Regularization parameter
    output : str
        'value and gradient', 'value' or 'gradient'
    regularization_mode : str
        'L1', 'L2', 'sigmoid'
        Type of regularization.

    Returns
    -------
    c : float
        Cost function value.

    """
    # Dimensions. Help variables.
    n_w = len(z)
    n_imp = np.shape(hyb)[0]
    n_b = len(eb)
    # Number of data points to fit to.
    m = n_imp*n_imp*n_w
    assert hyb.size == m
    # Convert hopping parameters to physical shape.
    v = unroll(p, n_b, n_imp)
    # Model hybridization functions.
    hyb_model = get_hyb(z, eb, v)
    # Difference between original and model hybridization functions
    diff = hyb_model - hyb
    if only_imag_part:
        # Consider only imaginary part of the hybrization functions.
        diff = diff.imag
        # Loss values
        loss = 1/2*diff**2
    else:
        # Loss values
        loss = 1/2*np.abs(diff)**2
    # Cost function.
    # Cost function value,
    # sum over two impurity orbital indices and
    # one energy index.
    c = 1/m*np.sum(loss)
    # Add regularization terms
    if regularization_mode == 'L1':
        # L1-regularization
        c +=  gamma/len(p)*np.sum(np.abs(p))
    elif regularization_mode == 'L2':
        # L2-regularization
        c +=  gamma/(2*len(p))*np.sum(p**2)
    elif regularization_mode == 'sigmoid':
        # Regularization with derivative being the sigmoid function.
        # This acts as a smoothened version of L1-regularization.
        # Delta determine the sharpness of the sigmoid function.
        delta = 0.1
        c +=  gamma/len(p)*np.sum(delta*np.log(np.cosh(p/delta)))


    else:
        sys.exit('Regularization mode not implemented.')

    if output == 'value':
        # Return only the cost function value.
        return c

    # Calculate gradient here...
    if not only_imag_part:
        sys.exit(('Gradient for fit to complex hybridization '
                  + 'is not implemented yet...'))

    # Partial derivatives of the cost function with respect to the
    # real and imaginary part of the hopping parameters.
    dcdv_re = np.zeros((n_b, n_imp), dtype=np.float)
    dcdv_im = np.zeros((n_b, n_imp), dtype=np.float)
    if True:
        # Calculate the gradient using some numpy broadcasting trixs.
        # complex array(B,M)
        green_b = 1/(z-np.atleast_2d(eb).T)
        # Loop over all impurity orbitals
        for r in range(n_imp):
            # Sum over columns of the hybridization matrix,
            # not being equal to column r.
            # Also sum over rows of the hybridization matrix,
            # not being equal to row r.
            for j in list(range(r)) + list(range(r+1, n_imp)):
                # complex array(B,1)
                v_matrix = np.atleast_2d(v[:,j]).T
                # diff[r,j,:], real array(M)
                # Sum real array(B,M) along energy axis.
                dcdv_re[:,r] += np.sum(
                    diff[r,j,:]*np.imag(v_matrix*green_b), axis=1)
                dcdv_im[:,r] += np.sum(
                    diff[r,j,:]*np.imag(-1j*v_matrix*green_b), axis=1)
                # Sum real array(B,M) along energy axis.
                dcdv_re[:,r] += np.sum(
                    diff[j,r,:]*np.imag(v_matrix.conj()*green_b), axis=1)
                dcdv_im[:,r] += np.sum(
                    diff[j,r,:]*np.imag(1j*v_matrix.conj()*green_b), axis=1)
            # complex array(B,1)
            v_matrix = np.atleast_2d(v[:,r]).T
            # diff[r,r,:], real array(M)
            # Add contribution from case with i=j=r
            # Sum real array(B,M) along energy axis.
            dcdv_re[:,r] += np.sum(
                diff[r,r,:]*np.imag(2*v_matrix.real*green_b), axis=1)
            dcdv_im[:,r] += np.sum(
                diff[r,r,:]*np.imag(2*v_matrix.imag*green_b), axis=1)
    else:
        # Calculate the gradient without any numpy broadcasting trixs.
        # Loop over all impurity orbitals
        for r in range(n_imp):
            # Sum over columns of the hybridization matrix,
            # not being equal to column r.
            for j in list(range(r)) + list(range(r+1, n_imp)):
                # Loop over all bath states
                for b in range(n_b):
                    # Sum over energies
                    dcdv_re[b,r] += np.sum(diff[r,j,:]*np.imag(v[b,j]/(z-eb[b])))
                    dcdv_im[b,r] += np.sum(diff[r,j,:]*np.imag(-1j*v[b,j]/(z-eb[b])))
            # Sum over rows of the hybridization matrix,
            # not being equal to row r.
            for i in list(range(r)) + list(range(r+1, n_imp)):
                # Loop over all bath states
                for b in range(n_b):
                    # Sum over energies
                    dcdv_re[b,r] += np.sum(diff[i,r,:]*np.imag(v[b,i].conj()/(z-eb[b])))
                    dcdv_im[b,r] += np.sum(diff[i,r,:]*np.imag(1j*v[b,i].conj()/(z-eb[b])))
            # Add contribution from case with i=j=r
            # Loop over all bath states
            for b in range(n_b):
                # Sum over energies
                dcdv_re[b,r] += np.sum(diff[r,r,:]*np.imag(2*v[b,r].real/(z-eb[b])))
                dcdv_im[b,r] += np.sum(diff[r,r,:]*np.imag(2*v[b,r].imag/(z-eb[b])))

    # Divide with normalization factor
    dcdv_re /= m
    dcdv_im /= m
    # Complex-valued matrix.
    dcdv = dcdv_re + 1j*dcdv_im
    # Convert to real-valued vector.
    dcdp = inroll(dcdv)
    # Add regularization contribution to the gradient.
    if regularization_mode == 'L1':
        # L1-regularization
        dcdp +=  gamma/len(p)*np.sign(p)
    elif regularization_mode == 'L2':
        # L2-regularization
        dcdp +=  gamma/len(p)*p
    elif regularization_mode == 'sigmoid':
        dcdp += gamma/len(p)*np.tanh(p/delta)
    else:
        sys.exit('Regularization mode not implemented.')


    if output == 'gradient':
        return dcdp
    elif output == 'value and gradient':
        return c, dcdp
    else:
        sys.exit('Output option not possible.')


    #return np.sum(np.abs(p-4))



def merge_ebs(ebs):
    """
    Returns the bath energies, merged into a 1d array.

    Parameters
    ----------
    ebs : tuple(K)
        Bath energies, for each energy window.
        Each element contains an array of bath states.

    Returns
    -------
    eb : array
        All the bath states as a one dimensional array.

    """
    eb = np.hstack(ebs)
    #eb = ebs.flatten()
    return eb

def merge_vs(vs):
    """
    Returns the hopping parameters, merged into a 2d array.

    Parameters
    ----------
    vs : tuple(K)
        Hopping parameters, for each energy window.
        Each element contains an array(B,N) of hopping parameters.

    Returns
    -------
    v : array
        All the hopping parameters as a two dimensional array(Btot, N),
        where Btot is the number of all the bath states.
    """
    #v = vs.reshape(vs.shape[0]*vs.shape[1], vs.shape[-1])
    v = np.vstack(vs)
    return v
