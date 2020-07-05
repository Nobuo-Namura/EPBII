# -*- coding: utf-8 -*-
"""
EGO_main.py
Copyright (c) 2020 Nobuo Namura
This code is released under the MIT License.

This Python code is for the EPBII/EIPBII infill criteria published in the following article:
N. Namura, K. Shimoyama, and S. Obayashi, "Expected Improvement of Penalty-based Boundary 
Intersection for Expensive Multiobjective Optimization," IEEE Transactions on Evolutionary 
Computation, vol. 21, no. 6, pp. 898-913, 2017.
Please cite the article if you use the code.

This code was developed with Python 3.6.5.
The original code used in the article had been implemented with Fortran.
This Python code is a converted version of it, and some results may differ from the article.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functools
import time
import shutil
from pyDOE import lhs

from kriging import Kriging
import test_problem
from initial_sample import generate_initial_sample
import indicator



#======================================================================
if __name__ == "__main__":
    """Constraints are unavailable in this version"""
    
    """=== Edit from here ==========================================="""
    func_name = 'ZDT3'        # Test problem name in test_problem.py
    nx = 8                    # Number of design variables (>=1)
    nf = 2                    # Number of objective functions (>=2)
    nh = 100                  # Division number for reference vector generation on hyper plane (recommended: 100, 20, and 10 for 2, 3, and 4 objectives, respectively)
    nhin = 0                  # Division number for reference vector generation on inner hyper plane (>=0)
    n_add = 5                 # Number of additional sample points at each iteration (>=1)
    max_iter = 5              # Number of EGO iteration (>=1)
    ntrial = 1                # Number of independent run with different initial samples (>=1)
    CRITERIA = 'EPBII'        # EPBII or EIPBII
    MIN = np.full(nf,True)    # True=Minimization, False=Maximization
    NOISE = np.full(nf,False) # Use True if functions are noisy (Griewank, Rastrigin, DTLZ1, etc.)
    PLOT = True               # True=Plot the results
    GENE = True               # True=Generate initial sample with LHS, False=Read files
    ns = nx*11-1              # Number of initial sample points when GENE=True (>=2)
    xmin = np.full(nx, 0.0)   # Lower bound of design sapce
    xmax = np.full(nx, 1.0)   # Upper bound of design sapce
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    fname_indicator = 'indicators'
    path_IGD_ref = current_dir + '/IGD_ref'
    """=== Edit End ================================================="""
    
    #Initial sample
    problem = functools.partial(eval('test_problem.'+func_name), nf=nf)
    if GENE:
        generate_initial_sample(func_name, nx, nf, ns, ntrial, xmin, xmax, current_dir, fname_design_space, fname_sample)
    f_design_space = current_dir + '/' + fname_design_space + '.csv'
    igd_ref = np.loadtxt(path_IGD_ref + '/' + func_name + 'f' + str(nf) + '.csv', delimiter=',')
    gp = Kriging(MIN=MIN, CRITERIA=CRITERIA, n_add=n_add, pbi_theta=1.0, nh=nh, nhin=nhin)
    
    #Preprocess for RMSE
    if nx == 2:
        ndiv = 101
        x_rmse0 = np.zeros([ndiv**2, nx])
        for i in range(101):
            for j in range(101):
                x_rmse0[i*ndiv+j,0] = float(i)/float(ndiv-1)
                x_rmse0[i*ndiv+j,1] = float(j)/float(ndiv-1)
    else:
        x_rmse0 = np.random.uniform(size=[10000, nx])

    #Independent run
    print('EGO')
    for itrial in range(1,ntrial+1,1):
        #Preprocess
        print('trial '+ str(itrial))
        f_sample = current_dir + '/' + fname_sample + str(itrial) + '.csv'
        gp.read_sample(f_sample)
        gp.normalize_x(f_design_space)
        x_rmse = gp.xmin + (gp.xmax-gp.xmin)*x_rmse0
        rmse = np.zeros([max_iter, gp.nf + gp.ng])
        igd = np.zeros(max_iter+1)
        times = []
        rank = gp.pareto_ranking(gp.f, gp.g)
        igd[0] = indicator.igd_history(gp.f[rank==1.0], igd_ref)
        f_indicator = current_dir + '/' + fname_indicator + str(itrial) +'.csv'
        with open(f_indicator, 'w') as file:
            data = ['iteration', 'samples', 'time', 'IGD']
            for i in range(gp.nf + gp.ng):
                data.append('RMSE'+str(i+1))
            data = np.array(data).reshape([1,len(data)])
            np.savetxt(file, data, delimiter=',', fmt = "%s")
        f_sample_out =  current_dir + '/' + fname_sample + str(itrial) + '_out.csv'
        shutil.copyfile(f_sample, f_sample_out)
        
        #Main loop for EGO
        for itr in range(max_iter):
            try:
                times.append(time.time())
                print('=== Iteration = '+str(itr)+', Number of sample = '+str(gp.ns)+' ======================')
                
                #Kriging and infill criterion
                gp.kriging_training(theta0 = 3.0, npop = 500, ngen = 500, mingen=0, STOP=True, NOISE=NOISE)
                x_add = gp.kriging_infill(PLOT=False)
                times.append(time.time())

                #RMSE
                for ifg in range(gp.nf + gp.ng):
                    gp.nfg = ifg
                    rmse[itr, ifg] = indicator.rmse_history(x_rmse, problem, gp.kriging_f, ifg)

                #Add sample points
                for i_add in range(gp.n_add):
                    f_add = problem(x_add[i_add])
                    gp.add_sample(x_add[i_add],f_add)
                
                #IGD and file output
                with open(f_indicator, 'a') as file:
                    data = np.hstack([itr, gp.ns-gp.n_add, times[-1]-times[-2], igd[itr], rmse[itr, :]])
                    np.savetxt(file, data.reshape([1,len(data)]), delimiter=',')
                with open(f_sample_out, 'a') as file:
                    data = np.hstack([gp.x[-gp.n_add:,:], gp.f[-gp.n_add:,:], gp.g[-gp.n_add:,:]])
                    np.savetxt(file, data, delimiter=',')
                rank = gp.pareto_ranking(gp.f, gp.g)
                igd[itr+1] = indicator.igd_history(gp.f[rank==1.0], igd_ref)
                if itr == max_iter-1:
                    with open(f_indicator, 'a') as file:
                        data = np.array([itr+1, gp.ns, 0.0, igd[itr+1]])
                        np.savetxt(file, data.reshape([1,len(data)]), delimiter=',')
                
                #Visualization
                if PLOT:
                    rank = gp.pareto_ranking(gp.f, gp.g)
                    f_pareto = gp.f[rank==1.0]
                    if len(MIN)==2:
                        plt.figure('test 2D Objective-space '+func_name+' at '+str(itr)+'-th iteration')
                        plt.plot(gp.f[:-gp.n_add,0], gp.f[:-gp.n_add,1], '.', c='black')
                        plt.plot(gp.f[-gp.n_add:,0], gp.f[-gp.n_add:,1], '.', c='magenta')
                        plt.plot(gp.utopia[0], gp.utopia[1], '+', c='black')
                        plt.plot(gp.nadir[0], gp.nadir[1], '+', c='black')
                        plt.plot(gp.refpoint[:,0], gp.refpoint[:,1], '.', c='blue',marker='+')
                        plt.scatter(gp.f_candidate[:,0],gp.f_candidate[:,1],c=gp.fitness_org,cmap='jet',marker='*')
                        plt.scatter(gp.f_ref[:,0],gp.f_ref[:,1],c='grey',s=1,marker='*')
                        plt.show(block=False)
                        title = current_dir + '/2D_Objective_space_'+func_name+'_at_'+str(itr)+'th_iteration_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title, dpi=300)
                        plt.close()
                        
                        plt.figure('solutions on 2D Objective-space '+func_name+' at '+str(itr)+'-th iteration')
                        plt.scatter(igd_ref[:,0],igd_ref[:,1],c='green',s=1)
                        plt.scatter(f_pareto[:,0],f_pareto[:,1],c='blue',s=20,marker='o')
                        title = current_dir + '/Optimal_solutions_'+func_name+'_at_'+str(itr)+'th_iteration_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title)
                        plt.close()
                        
                    elif len(MIN)==3:
                        fig = plt.figure('3D Objective-space '+func_name+' at '+str(itr)+'-th iteration')
                        ax = Axes3D(fig)
                        ax.scatter3D(gp.f[-gp.n_add:,0],gp.f[-gp.n_add:,1],gp.f[-gp.n_add:,2],c='red',marker='^')
                        ax.scatter3D(f_pareto[:,0],f_pareto[:,1],f_pareto[:,2],c='blue',marker='+')
                        ax.scatter3D(gp.f_candidate[:,0],gp.f_candidate[:,1],gp.f_candidate[:,2],c=gp.fitness_org,cmap='jet',marker='*')
                        ax.scatter3D(gp.f_ref[:,0],gp.f_ref[:,1],gp.f_ref[:,2],c='grey',s=1,marker='*')
                        title = current_dir + '/3D_Objective_space_'+func_name+'_at_'+str(itr)+'th_iteration_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title)
                        plt.close()
                        
                        fig2 = plt.figure('solutions on 3D Objective-space '+func_name+' at '+str(itr)+'-th iteration')
                        ax2 = Axes3D(fig2)
                        ax2.scatter3D(igd_ref[:,0],igd_ref[:,1],igd_ref[:,2],c='green',s=1)
                        ax2.scatter3D(f_pareto[:,0],f_pareto[:,1],f_pareto[:,2],c='blue',s=20,marker='o')
                        title = current_dir + '/Optimal_solutions_'+func_name+'_at_'+str(itr)+'th_iteration_in_'+str(itrial)+'-th_trial.png'
                        plt.savefig(title)
                        plt.close()
            except:
                break
