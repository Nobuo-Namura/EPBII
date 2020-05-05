# -*- coding: utf-8 -*-
"""
initial_sample.py
Copyright (c) 2020 Nobuo Namura
This code is released under the MIT License.
"""

import numpy as np
import pandas as pd
import functools
from pyDOE import lhs
import test_problem

#======================================================================
def generate_initial_sample(func_name, nx, nf, ns, ntrial, xmin, xmax, current_dir, fname_design_space, fname_sample):
    func = functools.partial(eval('test_problem.'+func_name), nf=nf)
    df_design_space = pd.DataFrame()
    df_design_space['min'] = xmin
    df_design_space['max'] = xmax
    df_design_space.to_csv(current_dir+'/'+fname_design_space+'.csv', index=None)
    
    print('Initial sample generation')
    for itrial in range(ntrial):
        print('trial '+ str(itrial+1))
        x = lhs(nx, samples=ns, criterion='cm',iterations=1000)
        x = xmin + x*(xmax - xmin)
        
        f = np.zeros((ns,nf))
        for i in range(ns):
            f[i,:] = func(x[i,:])
        
        df = pd.DataFrame()
        for i in range(nx):
            df['x'+str(i+1)] = x[:,i]
        for i in range(nf):
            df['f'+str(i+1)] = f[:,i]
        
        fname = fname_sample+str(itrial+1)+'.csv'
        df.to_csv(fname,index=None)

#======================================================================
if __name__ == "__main__":
    func_name = 'ZDT3'
    nx = 8
    nf = 2
    ns = nx*11-1
    ntrial = 1
    xmin = np.full(nx, 0.0)
    xmax = np.full(nx, 1.0)
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    generate_initial_sample(func_name, nx, nf, ns, ntrial, xmin, xmax, current_dir, fname_design_space, fname_sample)