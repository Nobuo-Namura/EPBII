# -*- coding: utf-8 -*-
"""
indicator.py
Copyright (c) 2020 Nobuo Namura
This code is released under the MIT License.
"""

import numpy as np
from scipy.spatial import distance

#======================================================================
def rmse_history(x_rmse, problem, func, nfg=0):
    rmse = 0.0
    for x in x_rmse:
        rmse += (problem(x)[nfg] - func(x))**2.0
    rmse = np.sqrt(rmse/float(len(x_rmse[:,0])))
    return rmse

#======================================================================
def igd_history(f, igd_ref):
    dist = distance.cdist(f, igd_ref)
    igd = np.mean(np.min(dist,axis=0))

    return igd