#!/usr/bin/env python3

"""

soc
===

This module contains functions which are useful to
process spin orbit coupling (SOC) data generated from by the  RSPt software.

"""

import numpy as np


def represents_int(s):
    """
    Return boolean about whether it is possible to
    convert input parameter to an int.

    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_core_energies(path):
    """
    Return all the core energies stored in RSPt out file.

    path - path to RSPt out file to parse.
    """
    with open(path, 'r') as f:
        data = f.readlines()
    its = []  # indices for the different types
    ts = []  # types
    # For each type, find first row about core energies
    for i, row in enumerate(data):
        if 'type:' in row:
            its.append(i)
            t = int(row.split()[1])
            ts.append(t)
    es = {}  # energies for the different types
    # Loop over all types
    for it, t in zip(its, ts):
        # First row containing energies
        i = it + 4
        es[t] = []
        while represents_int(data[i].split()[0]):
            es[t].append(float(data[i].split()[2]))
            i += 1
        es[t] = np.array(es[t])
    return es
