# -*- coding: utf-8 -*-
"""
Kriging.py
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
import sys
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from decimal import Decimal, getcontext
from scipy.special import comb
from scipy.stats import norm
from scipy.spatial import distance
import functools
from sklearn.cluster import KMeans
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D

from EA_in_DEAP.nsga2 import NSGA2
from singlega import SingleGA as SGA
import test_problem
from initial_sample import generate_initial_sample

#======================================================================
class Kriging:
#======================================================================
    def __init__(self, MIN=[True], CRITERIA='EST', n_add = 5, pbi_theta=1.0, nh=10, nhin=0):
        self.MIN = np.array(MIN)
        self.CRITERIA = CRITERIA
        self.LU_error = 0.1
        if self.CRITERIA == 'EPBII' or self.CRITERIA == 'EIPBII':
            self.n_add = n_add
            self.pbi_theta =  pbi_theta
            self.nh = nh
            self.nhin = nhin
            self.nrand = 100
            self.epsilon = 0.01
            self.tiny = 1.0e-20
            self.dist_threshold = 1.0e-8
            self.uni_rand = lhs(len(self.MIN), samples=self.nrand, criterion='cm',iterations=100)
            self.generate_refvec(nh=self.nh, nhin=self.nhin, PLOT=False)
        else:
            pass
        self.nfg = 0

#======================================================================
    def read_sample(self, f_sample):
        df_sample = pd.read_csv(f_sample)
        
        self.ns = len(df_sample)
        self.nx = sum(df_sample.columns.str.count('x'))
        self.nf = sum(df_sample.columns.str.count('f'))
        self.ng = sum(df_sample.columns.str.count('g'))
        
        self.x = np.asarray(df_sample.iloc[:,:self.nx])
        self.f = np.asarray(df_sample.iloc[:,self.nx:self.nx+self.nf])
        self.g = np.asarray(df_sample.iloc[:,self.nx+self.nf:])

#======================================================================
    def normalize_x(self, f_design_space):
        df_design_space = pd.read_csv(f_design_space)
        
        self.xmin = df_design_space['min'].values
        self.xmax = df_design_space['max'].values
        
        self.x0 = (self.x - self.xmin)/(self.xmax - self.xmin)

#======================================================================
    def kriging_training(self, theta0 = 3.0, npop = 500, ngen = 500, mingen = 0, STOP=True, NOISE=[False]):
        self.theta = np.zeros((self.nf+self.ng, self.nx+1))
        func = self.likelihood
        getcontext().prec = 28#56
        
        for i in range(self.nf + self.ng):
            print('--- '+str(i+1)+'-th function estimation -------------------')
            self.nfg = i
            if NOISE[i]:
                theta_min = np.full(self.nx+1, -4.0)
                theta_max = np.full(self.nx+1, theta0)
                theta_min[-1] = -20.0
                theta_max[-1] = 1.0
            else:
                theta_min = np.full(self.nx, -4.0)
                theta_max = np.full(self.nx, theta0)
            sga = SGA(func, theta_min, theta_max, npop=npop, ngen=ngen, mingen=mingen, \
                      MIN=False, STOP=STOP, PRINT=True, INIT=False, \
                      pcross=0.9, pmut=1.0/len(theta_min), eta_c=10.0, eta_m=20.0)
            Ln, theta = sga.optimize()
            if NOISE[i]:
                self.theta[i,:] = 10.0**theta
            else:
                self.theta[i,:self.nx] = 10.0**theta
        
        self.kriging_construction()

#======================================================================
    def likelihood(self, theta0):
        theta = 10.0**theta0
        R, detR, mu, sigma, xtheta = self.corr_matrix(theta, self.nfg)
        if detR > 0.0 and sigma > 0.0:
            Ln = -0.5*float(Decimal(self.ns*np.log(sigma)) + detR.ln())
        else:
            Ln = -1.e+20
        return Ln

#======================================================================
    def corr_matrix(self, theta, nfg):
        xtheta = np.sqrt(theta[:self.nx])*self.x0
        R = np.exp(-distance.cdist(xtheta, xtheta)**2.0)
        if self.nx < len(theta):
            R += np.diag(np.full(len(R),theta[-1]))
        ones = np.ones(self.ns)
        Ri = linalg.lu_factor(R)
        detR = np.prod([Decimal(Ri[0][i,i]) for i in range(len(Ri[0]))]) #Higher accuray (than linalg.det) is needed for better approximation
        error = np.max(np.abs(linalg.lu_solve(Ri,R) - np.identity(len(R))))
        if detR > 0.0 and error < self.LU_error:
            if nfg < self.nf:
                mu = np.dot(ones, linalg.lu_solve(Ri,self.f[:,nfg]))/np.dot(ones, linalg.lu_solve(Ri,ones))
                fmu = self.f[:,nfg] - mu
                sigma =  np.dot(fmu, linalg.lu_solve(Ri,fmu))/self.ns
            else:
                mu = np.dot(ones, linalg.lu_solve(Ri,self.g[:,nfg-self.nf]))/np.dot(ones, linalg.lu_solve(Ri,ones))
                fmu = self.g[:,nfg-self.nf] - mu
                sigma =  np.dot(fmu, linalg.lu_solve(Ri,fmu))/self.ns
        else:
            detR = 0.0
            mu = 0.0
            sigma = 1.0
        return R, detR, mu, sigma, xtheta

#======================================================================
    def kriging_construction(self):
        self.mu = np.zeros(self.nf + self.ng)
        self.sigma = np.zeros(self.nf + self.ng)
        self.Ri = []
        self.Rifm = np.zeros([self.ns, self.nf + self.ng])
        self.Ri1 = np.zeros([self.ns, self.nf + self.ng])
        self.xtheta = np.zeros([self.ns, self.nx, self.nf + self.ng])
        
        for i in range(self.nf + self.ng):
            R, detR, self.mu[i], self.sigma[i], self.xtheta[:,:,i] = self.corr_matrix(self.theta[i,:], i)
            Ri = linalg.lu_factor(R)
            self.Ri.append(Ri)
            self.Ri1[:,i] = linalg.lu_solve(Ri, np.ones(self.ns))
            if i < self.nf:
                self.Rifm[:,i] = linalg.lu_solve(Ri, self.f[:,i]-self.mu[i])
            else:
                self.Rifm[:,i] = linalg.lu_solve(Ri, self.g[:,i-self.nf]-self.mu[i])
        return

#======================================================================
    def kriging_estimation(self, xs, nfg=-1):
        if nfg >= 0:
            self.nfg = nfg
        xs0 = (xs - self.xmin)/(self.xmax - self.xmin)
        xstheta = np.sqrt(self.theta[self.nfg,:self.nx])*xs0
        r = np.exp(-distance.cdist(xstheta.reshape([1,len(xstheta)]), self.xtheta[:,:,self.nfg])**2.0).reshape(self.ns)
        f = self.mu[self.nfg] + np.dot(r, self.Rifm[:,self.nfg])
        Rir = linalg.lu_solve(self.Ri[self.nfg], r)
        ones = np.ones(len(self.Rifm[:,0]))
        s = self.sigma[self.nfg]*(1.0 - np.dot(r,Rir) + ((1.0-np.dot(ones,Rir))**2.0)/np.dot(ones,self.Ri1[:,self.nfg]))
        s = np.sqrt(np.max([s, 0.0]))
        return f, s

#======================================================================
    def expected_improvement(self, fref, f, s):
        if s > 0.0:
            if self.MIN[self.nfg]:
                y = (fref - f)/s
            else:
                y = (f - fref)/s
            cdf = 0.5*(1.0 - np.erf(-y/np.sqrt(2.0)))
            pdf = 1.0/np.sqrt(2.0*np.pi)*np.exp(-0.5*y**2.0)
            ei = s*(y*cdf + pdf)   
        else:
            ei = 0.0
        return ei

#======================================================================
    def kriging_candidate(self, xs):
        f, s = self.kriging_estimation(xs, nfg=self.nfg)
        if self.CRITERIA == 'EST':
            return f
        elif self.CRITERIA == 'RMSE':
            return s
        elif self.CRITERIA == 'EI':
            if self.MIN[self.nfg]:
                fref = np.min(self.f[:,self.nfg])
            else:
                fref = np.max(self.f[:,self.nfg])
            ei = self.expected_improvement(fref, f, s)
            return ei
        else:
            return s

#======================================================================
    def kriging_f(self, xs):
        f, s = self.kriging_estimation(xs, nfg=self.nfg)
        return f

#======================================================================
    def kriging_multiobjective_f(self, xs):
        f = np.zeros(self.nf)
        for i in range(self.nf):
            f[i], s = self.kriging_estimation(xs, nfg=i)
        return f

#======================================================================
    def kriging_s(self, xs):
        f, s = self.kriging_estimation(xs, nfg=self.nfg)
        return s

#======================================================================
    def kriging_infill(self, PLOT=False):
        if self.CRITERIA == 'EPBII' or self.CRITERIA == 'EIPBII':
            self.utopia_nadir_on_kriging(PLOT=PLOT)
            self.reference_pbi()
            
            self.f_candidate = np.zeros([self.nref,self.nf])
            self.x_candidate = np.zeros([self.nref,self.nx])
            self.epbii = np.zeros(self.nref)
            for iref in range(self.nref):
                f_opt0 = (self.f_opt - self.utopia)/(self.nadir - self.utopia)
                if self.CRITERIA == 'EIPBII':
                    f_opt0 = -1.0 + f_opt0
                i_opt = np.argmax(np.abs(np.dot(f_opt0/np.reshape(np.linalg.norm(f_opt0,axis=1),[-1,1]), self.refvec[iref,:])))
                print('--- '+str(iref+1)+'-th reference vector --------------------')
                if self.CRITERIA == 'EPBII':
                    func = functools.partial(self.kriging_epbii, kref=iref)
                else:
                    func = functools.partial(self.kriging_eipbii, kref=iref)
                sga = SGA(func, self.xmin, self.xmax, npop=200, ngen=50, MIN=False, STOP=True, PRINT=False, INIT=True, \
                          pcross=0.9, pmut=1.0/self.nx, eta_c=10.0, eta_m=20.0)
                self.epbii[iref], self.x_candidate[iref,:] = sga.optimize(x_init=self.x_opt[i_opt,:])
                for iobj in range(self.nf):
                    self.f_candidate[iref,iobj], s = self.kriging_estimation(self.x_candidate[iref,:], nfg=iobj)
                    
            self.rank = self.pareto_ranking(self.f_candidate, np.ones([self.nref,1]))
            self.fitness = self.epbii/(self.nich_count*self.rank)
            dist = np.min(distance.cdist(self.x, self.x_candidate), axis=0)
            self.fitness = np.where(dist>self.dist_threshold, self.fitness, -1.0e20)
            x_add = np.zeros([self.n_add,self.nx])
            self.fitness_org = self.fitness.copy()
            for i_add in range(self.n_add):
                i_candidate = np.where((self.refvec_cluster==i_add) & \
                                       (self.fitness==np.max(self.fitness[self.refvec_cluster==i_add])))[0][0]
                x_add[i_add,:] = self.x_candidate[i_candidate,:]
                self.nich[i_candidate] += 1
                self.nich_count = np.array([np.sum(self.nich/self.normalized_refvec_distance[i,:]) for i in range(self.nref)])
                self.fitness = np.where(dist>self.dist_threshold, self.epbii/(self.nich_count*self.rank), -1.0e20)
            # remove duplicated samples
            dist = distance.cdist(x_add, x_add)
            for i_add in range(self.n_add):
                if np.min(np.hstack([dist[i_add,:i_add], dist[i_add,i_add+1:]])) <= self.dist_threshold:
                    print('a sample point candidate was changed')
                    flag = True
                    # replaced from the same cluster
                    for j in range(1, len(self.fitness[self.refvec_cluster==i_add])):
                        i_candidate = np.argsort(self.fitness[self.refvec_cluster==i_add])[::-1][j]
                        x_add[i_add,:] = self.x_candidate[self.refvec_cluster==i_add][i_candidate,:]
                        dist = distance.cdist(x_add, x_add)
                        if np.min(np.hstack([dist[i_add,:i_add], dist[i_add,i_add+1:]])) > self.dist_threshold:
                            flag = False
                            break
                    # replaced from other clusters
                    if flag:
                        for j in range(len(self.fitness)):
                            i_candidate = np.argsort(self.fitness)[::-1][j]
                            x_add[i_add,:] = self.x_candidate[i_candidate,:]
                            dist = distance.cdist(x_add, x_add)
                            if np.min(np.hstack([dist[i_add,:i_add], dist[i_add,i_add+1:]])) > self.dist_threshold:
                                flag = False
                                break            
            return x_add

#======================================================================
    def generate_refvec(self, nh=100, nhin=0, PLOT=False):
        def generate_vector(refvec, vector, ih, nh, nf, iref, i):
            if i == nf-1:
                vector[i] = float(ih)/float(nh)
                refvec[iref,:] = vector.copy()
                iref += 1
            else:
                for j in range(ih+1):
                    vector[i] = float(j)/float(nh)
                    refvec, iref = generate_vector(refvec, vector, ih-j, nh, nf, iref, i+1)
            return refvec, iref
    
        self.nf = len(self.MIN)
        self.nh = nh
        self.nhin = nhin
        self.nref = int(comb(self.nh+self.nf-1, self.nf-1))
        self.ref_theta = 1.0/(np.tan(0.25*np.pi/self.nh))
        if nhin > 0:
            self.nrefin = int(comb(self.nhin+self.nf-1, self.nf-1))
            self.nref += self.nrefin
        refvec = np.zeros([self.nref, self.nf])
        vector = np.zeros(self.nf)
        iref= 0 
        refvec, iref = generate_vector(refvec, vector, self.nh, self.nh, self.nf, iref, 0)
        
        if self.nhin > 0:
            inref = iref
            refvec, iref = generate_vector(refvec, vector, self.nhin, self.nhin, self.nf, iref, 0)
            tau = 0.5
            refvec[inref:,:] = (1.0-tau)/float(self.nf) + tau*refvec[inref:,:]
        
        self.refvec = refvec/np.reshape(np.linalg.norm(refvec,axis=1),[-1,1]) #normalize
        if self.CRITERIA == 'EIPBII':
            self.refvec = -self.refvec
        
        km = KMeans(n_clusters=self.n_add, init='k-means++', n_init=100, max_iter=10000)
        self.refvec_cluster = km.fit_predict(refvec) #clustering on hyperplane
        
        self.normalized_refvec_distance = np.zeros([self.nref,self.nref])
        for j in range(self.nref):
            for i in range(self.nref):
                self.normalized_refvec_distance[i,j] = np.linalg.norm(refvec[i,:] - refvec[j,:])
        self.dmin = np.min(self.normalized_refvec_distance[self.normalized_refvec_distance>0])
        self.normalized_refvec_distance = self.normalized_refvec_distance/self.dmin + 1.0

        if PLOT and self.nf==3:
            fig = plt.figure('refvec-3D')
            ax = Axes3D(fig)
            for i in range(self.n_add):
                ax.scatter3D(refvec[self.refvec_cluster==i,0],refvec[self.refvec_cluster==i,1],refvec[self.refvec_cluster==i,2])
        return

#======================================================================
    def utopia_nadir_on_kriging(self, PLOT=False):
        self.utopia = np.zeros(self.nf)
        self.nadir = np.ones(self.nf)
        
        weight = 1.0 - 2.0*self.MIN.astype(np.float)
        self.f_opt, self.x_opt = NSGA2(self.kriging_multiobjective_f, self.xmin.tolist(), self.xmax.tolist(), nx=self.nx, \
                                       weights=weight.tolist(), npop=500, ngen=100, p_cross=0.9, eta_cross=10.0, eta_mut=20.0, \
                                       PRINT=True, PLOT=PLOT)

        rank = self.pareto_ranking(self.f_opt, np.ones([len(self.f_opt),1]))
        f_epsilon = np.zeros(self.nf)
        if len(rank[rank==1.0]) >= self.nf:
            self.f_opt = self.f_opt[rank==1.0]
            self.x_opt = self.x_opt[rank==1.0]
            f_min = np.min(self.f_opt, axis=0)
            f_max = np.max(self.f_opt, axis=0)
            f_opt0 = (self.f_opt - f_min)/(f_max - f_min)
            #remove weak Pareto-optimum
            flag = np.full(len(f_opt0), True)
            nearest = np.zeros(len(f_opt0), dtype=int)
            for i in range(len(f_opt0)):
                distmin = 1.0e+20
                for j in range(len(f_opt0)):
                    dist = np.linalg.norm(f_opt0[i,:] - f_opt0[j,:])
                    if i!=j and dist < distmin:
                        distmin = dist
                        nearest[i] = j
                diff = np.abs(f_opt0[i,:] - f_opt0[nearest[i],:])
                diff_min = np.min(diff)
                diff_max = np.max(diff)
                if diff_max != 0.0:
                    ratio = diff_min/diff_max
                else:
                    ratio = 0.0
                if ratio < self.epsilon:
                    flag[i] = False
            if len(self.f_opt[flag]) >= self.nf:
                self.f_ref = self.f_opt[flag]
            else:
                self.f_ref = self.f_opt
        elif len(self.f) >= self.nf:
            rank = self.pareto_ranking(self.f, self.g)
            f_opt = self.f[rank==1.0]
            if len(f_opt) >= self.nf:
                self.f_ref = f_opt
            else:
                rank = self.pareto_ranking(self.f, np.ones([self.nf,1]))
                f_opt = self.f[rank==1.0]
                if len(f_opt) >= self.nf:
                    self.f_ref = f_opt
                else:
                    self.f_ref = self.f
        else:
            print('normalization failed')
            sys.exit()
        
        f_min = np.min(self.f_ref, axis=0) - f_epsilon
        f_max = np.max(self.f_ref, axis=0) + f_epsilon
        minmax = 1.0 - 2.0*self.MIN.astype(np.float)
        self.utopia = np.where(minmax<0.0, f_min, f_max)
        self.nadir = np.where(minmax<0.0, f_max, f_min)
        print('Utopia: ', self.utopia)
        print('Nadir: ', self.nadir)
        
        return

#======================================================================
    def reference_pbi(self):
        rank = self.pareto_ranking(self.f, self.g)
        f0 = (self.f - self.utopia)/(self.nadir - self.utopia)
        sign = 1.0
        if self.CRITERIA == 'EIPBII':
            f0 = f0 - 1.0
            sign = -1.0
        
        pbi_sample = np.zeros([self.ns, self.nref])
        self.near_vector = np.zeros(self.ns, dtype=int)
        z = np.zeros(self.nf)
        for i in range(self.ns):
            distmin = 1.0e+20
            for j in range(self.nref):
                pbi_sample[i,j], d1, d2 = self.evaluate_pbi(z, f0[i,:], self.refvec[j,:], self.pbi_theta, sign=sign)
                if d2 < distmin:
                    distmin = d2
                    self.near_vector[i] = j
        
        self.nich = np.zeros(self.nref, dtype=int)
        if self.CRITERIA == 'EPBII':
            self.pbiref = 1.1e+20*np.ones(self.nref)
        else:
            self.pbiref = -1.1e+20*np.ones(self.nref)
        for i in range(self.ns):
            k = self.near_vector[i]
            if rank[i] == 1:
                self.nich[k] += 1
            if self.CRITERIA == 'EPBII' and pbi_sample[i,k] < self.pbiref[k]:
                self.pbiref[k] = pbi_sample[i,k]
            elif self.CRITERIA == 'EIPBII' and pbi_sample[i,k] > self.pbiref[k]:
                self.pbiref[k] = pbi_sample[i,k]
                
        self.nich_count = np.array([np.sum(self.nich/self.normalized_refvec_distance[i,:]) for i in range(self.nref)])
        if self.CRITERIA == 'EPBII':
            pbimax = np.max(self.pbiref[self.pbiref<1.0e20])
            self.pbiref = np.where(self.pbiref<1.0e20, self.pbiref, 1.1*pbimax)
            refpoint = self.pbiref.reshape([self.nref,1])*self.refvec
            self.refpoint = self.utopia + refpoint*(self.nadir - self.utopia)
        else:
            pbimin = np.min(self.pbiref[self.pbiref>-1.0e20])
            self.pbiref = np.where(self.pbiref>-1.0e20, self.pbiref, pbimin-0.1*np.abs(pbimin))
            refpoint = self.pbiref.reshape([self.nref,1])*self.refvec
            self.refpoint = self.utopia + (1.0 + refpoint)*(self.nadir - self.utopia)
        return

#======================================================================
    def kriging_epbii(self, xs, kref):
        f = np.zeros(self.nf)
        s = np.zeros(self.nf)
        z = np.zeros(self.nf)
        for i in range(self.nf):
            f[i], s[i] = self.kriging_estimation(xs, nfg=i)
        f0 = (f - self.utopia)/(self.nadir - self.utopia)
        s0 = np.max([s/(self.nadir - self.utopia), np.full(self.nf,self.tiny)],axis=0)
        #Territory
        terr, d1t, d2t = self.evaluate_pbi(z, f0, self.refvec[kref,:], self.ref_theta, sign=-1.0)
        if terr < 0.0:
            epbii = terr
        #EPBII
        else:
            fp0 = norm.ppf(self.uni_rand, loc=f0, scale=s0)
            pbis, d1s, d2s = self.evaluate_pbis(z, fp0, self.refvec[kref,:], self.pbi_theta)
            pbiis = np.max(np.vstack([self.pbiref[kref]-pbis, np.zeros(self.nrand)]),axis=0)
            epbii = np.mean(pbiis)
            #Accelerate convergence
            if epbii <= 0.0:
                pbi, d1, d2 = self.evaluate_pbi(z, f0, self.refvec[kref,:], self.pbi_theta)
                epbii = np.min([0.0, self.pbiref[kref]-pbi])
        return epbii

#======================================================================
    def kriging_eipbii(self, xs, kref):
        f = np.zeros(self.nf)
        s = np.zeros(self.nf)
        z = np.zeros(self.nf)
        for i in range(self.nf):
            f[i], s[i] = self.kriging_estimation(xs, nfg=i)
        f0 = -1.0 + (f - self.utopia)/(self.nadir - self.utopia)
        s0 = np.max([s/(self.nadir - self.utopia), np.full(self.nf,self.tiny)],axis=0)
        #Territory
        terr, d1t, d2t = self.evaluate_pbi(z, f0, self.refvec[kref,:], self.ref_theta, sign=-1.0)
        if terr < 0.0:
            iepbii = terr
        #EIPBII
        else:
            fp0 = norm.ppf(self.uni_rand, loc=f0, scale=s0)
            ipbis, d1s, d2s = self.evaluate_pbis(z, fp0, self.refvec[kref,:], self.pbi_theta, sign=-1.0)
            ipbiis = np.max(np.vstack([ipbis-self.pbiref[kref], np.zeros(self.nrand)]),axis=0)
            iepbii = np.mean(ipbiis)
            #Accelerate convergence
            if iepbii <= 0.0:
                ipbi, d1, d2 = self.evaluate_pbi(z, f0, self.refvec[kref,:], self.pbi_theta, sign=-1.0)
                iepbii = np.min([0.0, ipbi-self.pbiref[kref]])
        return iepbii

#======================================================================
    def evaluate_pbi(self, z, f, vector, theta, sign=1.0):
        d1 = np.dot(f - z, vector)
        d2 = np.linalg.norm(f - (z + d1*vector))
        pbi = d1 + sign*theta*d2
        
        return pbi, d1, d2

#======================================================================
    def evaluate_pbis(self, z, f, vector, theta, sign=1.0):
        d1 = np.dot(f - z, vector)
        d2 = np.linalg.norm(f - (z + np.dot(d1.reshape([len(d1),1]),vector.reshape([1,len(vector)]))), axis=1)
        pbi = d1 + sign*theta*d2
        
        return pbi, d1, d2

#======================================================================
    def pareto_ranking(self, f, g):
        ns = len(f[:,0])
        nf = len(f[0,:])
        rank = np.ones(ns)
        for i in range(ns):
            if all(g[i,:]>0):
                for j in range(ns):
                    if all(g[j,:]>0):
                        irank = 0
                        for iobj in range(nf):
                            if self.MIN[iobj] and f[i,iobj]>=f[j,iobj]:
                                irank += 1
                            elif (not self.MIN[iobj]) and f[i,iobj]<=f[j,iobj]:
                                irank += 1
                        if i!=j and irank == nf:
                            rank[i] += 1
            else:
                rank[i] = -1
        rank = np.where(rank>0, rank, np.max(rank)+1)
        return rank

#======================================================================
    def add_sample(self, x_add, fg_add):
        self.ns += 1
        xadd = np.reshape(x_add, [1,len(x_add)])
        fgadd = np.reshape(fg_add, [1,len(fg_add)])
        self.x = np.vstack([self.x, xadd])
        x0add = (xadd - self.xmin)/(self.xmax - self.xmin)
        self.x0 = np.vstack([self.x0, x0add])
        self.f = np.vstack([self.f, fgadd[:,:self.nf]])
        self.g = np.vstack([self.g, fgadd[:,self.nf:]])
        return

#======================================================================
if __name__ == "__main__":
        
    """=== Edit from here ==========================================="""
    func_name = 'sphere'      # Test problem name in test_problem.py
    nx = 2                    # Number of design variables
    nf = 1                    # Number of objective functions
    ns = 10                   # Number of initial sample points when GENE=True
    ntrial = 1                # Number of independent run with different initial samples
    MIN = np.full(nf,True)    # Minimization: True, Maximization: False
    NOISE = np.full(nf,False) # Use True if functions are noisy (Griewank, Rastrigin, DTLZ1, etc.)
    xmin = np.full(nx,-0.5)   # Lower bound of design sapce
    xmax = np.full(nx, 0.5)   # Upper bound of design sapce
    current_dir = '.'
    fname_design_space = 'design_space'
    fname_sample = 'sample'
    """=== Edit End ================================================="""


    f_sample = current_dir + '/' + fname_sample + '1.csv'
    f_design_space = current_dir + '/' + fname_design_space + '.csv'
    
    func = functools.partial(eval('test_problem.'+func_name), nf=nf)
    generate_initial_sample(func_name, nx, nf, ns, ntrial, xmin, xmax, current_dir, fname_design_space, fname_sample)
    
    gp = Kriging(MIN=MIN)
    gp.read_sample(f_sample)
    gp.normalize_x(f_design_space)
    gp.kriging_training(theta0 = 3.0, npop = 500, ngen = 500, mingen=0, STOP=True, NOISE=NOISE)
    
    if nx == 2:
        x = gp.xmin[0]+np.arange(0., 1.01, 0.01)*(gp.xmax[0]-gp.xmin[0])
        y = gp.xmin[1]+np.arange(0., 1.01, 0.01)*(gp.xmax[1]-gp.xmin[1])
        X, Y = np.meshgrid(x, y)
        F = X.copy()
        S = X.copy()
        gp.nfg = 0
        for i in range(len(X[:,0])):
            for j in range(len(X[0,:])):
                F[i,j], S[i,j] = gp.kriging_estimation(np.array([X[i,j],Y[i,j]]))
        plt.figure('Kriging')
        plt.plot(gp.x[:,0],gp.x[:,1],'o',c='black')
        plt.pcolor(X,Y,F,cmap='jet')
        plt.colorbar()
        plt.contour(X,Y,F,40,colors='black',linestyles='solid')
        plt.show()
    
    n_valid = 10000
    fs = np.zeros([n_valid,2])
    R2 = np.zeros(nf)
    x_valid = gp.xmin + np.random.rand(n_valid, nx)*(gp.xmax - gp.xmin)
    for i in range(nf):
        for j in range(n_valid):
            fs[j,0], ss = gp.kriging_estimation(x_valid[j,:], nfg=i)
            if nf > 1:
                fs[j,1] = func(x_valid[j,:])[i]
            else:
                fs[j,1] = func(x_valid[j,:])
        delt = fs[:,0]-fs[:,1]
        R2[i] = 1-(np.dot(delt,delt)/float(n_valid))/np.var(fs[:,1])
        plt.figure('objective'+str(i+1))
        plt.plot(fs[:,1], fs[:,0], '.')
    print(R2)

