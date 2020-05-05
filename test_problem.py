# -*- coding: utf-8 -*-
"""
test_problem.py
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

#======================================================================
def sphere(x, nf=1):
    x = np.array(x)
    f = np.dot(x,x)
    return f

#======================================================================
def DTLZ1(x, nf=3):
    x = np.array(x)
    
    g = 100.0*(float(len(x[nf-1:])) + np.sum((x[nf-1:]-0.5)**2.0 - np.cos(20.0*np.pi*(x[nf-1]-0.5))))
    f = np.full(nf, 0.5*(1.0+g))
    for i in range(nf):
        f[i] *= np.prod(x[:nf-i-1])
        if i > 0:
            f[i] *= 1.0-x[nf-i-1]
    
    return f

#======================================================================
def DTLZ2(x, nf=3):
    x = np.array(x)
    
    g = np.sum((x[nf-1:] - 0.5)**2.0)
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ3(x, nf=3):
    x = np.array(x)
    
    g = 100.0*(float(len(x[nf-1:])) + np.sum((x[nf-1:]-0.5)**2.0 - np.cos(20.0*np.pi*(x[nf-1]-0.5))))
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ4(x, nf=3):
    x = np.array(x)
    alpha = 100.0
    
    g = np.sum((x[nf-1:]-0.5)**2.0)
    f = np.full(nf, 1.0+g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]**alpha))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1]**alpha)
    
    return f

#======================================================================
def DTLZ7(x, nf=3):
    x = np.array(x)
    
    g = 1.0 + 9.0/float(len(x[nf-1:]))*np.sum(x[nf-1:])
    h = float(nf)
    f = np.zeros(nf)
    for i in range(nf-1):
        f[i] = x[i]
        h -= (1.0 + np.sin(3.0*np.pi*f[i]))*f[i]/(1.0 + g)
    f[-1] = (1.0 + g)*h
    
    return f

#======================================================================
def DTLZ2max1(x, nf=3):
    x = np.array(x)
    
    g = np.sum(1.0 - 4.0*(x[nf-1:]-0.5)**2.0)/float(len(x[nf-1:]))
    f = np.full(nf, g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ2max2(x, nf=3):
    x = np.array(x)
    x[:nf-1] = 0.25 + 0.5*x[:nf-1]
    
    g = np.sum(1.0 - 4.0*(x[nf-1:]-0.5)**2.0)/float(len(x[nf-1:]))
    f = np.full(nf, g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def DTLZ2max3(x, nf=3):
    x = np.array(x)
    x[:nf-1] = 0.25 + 0.5*x[:nf-1]
    
    g = np.sum(1.0 - (x[nf-1:]-0.5)**2.0 + (np.cos(4.0*np.pi*(x[nf-1:]-0.5))-1.0)/3.0)/float(len(x[nf-1:]))
    f = np.full(nf, g)
    for i in range(nf):
        f[i] *= np.prod(np.cos(0.5*np.pi*x[:nf-i-1]))
        if i > 0:
            f[i] *= np.sin(0.5*np.pi*x[nf-i-1])
    
    return f

#======================================================================
def ZDT1(x, nf=2):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0/float(len(x[1:]))*np.sum(x[1:])
    f[1] = g*(1.0 - np.sqrt(f[0]/g))
    
    return f

#======================================================================
def ZDT2(x, nf=2):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0/float(len(x[1:]))*np.sum(x[1:])
    f[1] = g*(1.0 - (f[0]/g)**2.0)
    
    return f

#======================================================================
def ZDT3(x, nf=2):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 9.0/float(len(x[1:]))*np.sum(x[1:])
    f[1] = g*(1.0 - np.sqrt(f[0]/g) - x[0]/g*np.sin(10.0*np.pi*x[0]))
    
    return f

#======================================================================
def ZDT4(x, nf=2):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = x[0]
    g = 1.0 + 10.0*float(len(x[1:])) + np.sum(x[1:]**2.0 - 10.0*np.cos(4.0*np.pi*x[1:]))
    f[1] = g*(1.0 - np.sqrt(f[0]/g))
    
    return f

#======================================================================
def ZDT6(x, nf=2):
    x = np.array(x)
    f = np.zeros(2)
    f[0] = 1.0 - np.exp(-4.0*x[0])*(np.sin(6.0*np.pi*x[0]))**6.0
    g = 1.0 + 9.0*(np.sum(x[1:])/float(len(x[1:])))**0.25
    f[1] = g*(1.0 - (f[0]/g)**2.0)
    
    return f

#======================================================================
def LZ08F1(x, nf=2):
    x = np.array(x)
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-x[0]**(0.5+3*(i-2)/(2*(nx-2))))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-x[0]**(0.5+3*(i-2)/(2*(nx-2))))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
def LZ08F2(x, nf=2):
    x = np.array(x)
    x[1:] = -1.0 + 2.0*x[1:]
    
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-np.sin(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-np.sin(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
def LZ08F3(x, nf=2):
    x = np.array(x)
    x[1:] = -1.0 + 2.0*x[1:]
    
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-0.8*np.cos(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-0.8*np.cos(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(2, nx+1, 2)])
    
    return f

#======================================================================
def LZ08F4(x, nf=2):
    x = np.array(x)
    x[1:] = -1.0 + 2.0*x[1:]
    
    f = np.zeros(2)
    nx = len(x)
    if np.mod(nx,2) == 0:
        n1 = (nx-2)/2
        n2 = n1 + 1
    else:
        n1 = (nx-1)/2
        n2 = n1
    f[0] = x[0] + 2.0/n1*np.sum([(x[i-1]-0.8*x[0]*np.cos(2.0*np.pi*x[0]+i*np.pi/(3*nx)))**2.0 for i in range(3, nx+1, 2)])
    f[1] = 1.0 - np.sqrt(x[0]) + 2.0/n2*np.sum([(x[i-1]-0.8*x[0]*np.sin(6.0*np.pi*x[0]+i*np.pi/nx))**2.0 for i in range(2, nx+1, 2)])
    
    return f
