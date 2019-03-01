#!/usr/bin/env python3

"""

slater
======

This module contains functions which are useful to
process Slater-Condon data generated from the RSPt software.

"""

import numpy as np
from math import pi
from numba import jit


# ----------------------------------------------------------
# Slater-Condon integrals
@jit
def get_Slater_F(r, f1, f2, k):
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
    >>> from rspt2spectra.slater import get_Slater_F
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
    dr[1:-1] = (r[2:] - r[:-2]) / 2.
    dr[0] = (r[1] - r[0]) / 2.
    dr[-1] = (r[-1] - r[-2]) / 2.
    n = len(r)
    s = 0
    for i in range(n):
        for j in range(n):
            s += (dr[i] * dr[j] * (r[i] * r[j] * f1[i] * f2[j]) ** 2
                  * min(r[i], r[j]) ** k
                  / max(r[i], r[j]) ** (k + 1))
    return s


@jit
def get_Slater_G(r, f1, f2, k):
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
    >>> from rspt2spectra.slater import get_Slater_G
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
    dr[1:-1] = (r[2:] - r[:-2]) / 2.
    dr[0] = (r[1] - r[0]) / 2.
    dr[-1] = (r[-1] - r[-2]) / 2.
    n = len(r)
    s = 0
    for i in range(n):
        for j in range(n):
            s += (dr[i] * dr[j] * (r[i] * r[j]) ** 2
                  * f1[i] * f2[i] * f1[j] * f2[j]
                  * min(r[i], r[j]) ** k
                  / max(r[i], r[j]) ** (k + 1))
    return s
