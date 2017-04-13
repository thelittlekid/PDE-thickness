#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:31:40 2017

@author: fantasie
"""

import numpy as np

def calculate_gradient(u, mode = 'central'):
    """
        Calculate the gradient of each point in u
    Args:
        u: the input array, size: d1 x d2 x ... x dn
        mode: mode of spatial difference: central, forward, backward

    Returns:
        grad: a list [u_1, u_2, ...], each element in the list corresponds to
            the partial derivative w.r.t a dimension, each element has the same
            dimension as the input u: d1 x d2 ... x dn
    """
    grad = [None] * u.ndim
    for n in xrange(u.ndim):
        # the shifted matrices, ignore the boundary error from cyclic shift
        u_p = np.roll(u, 1, axis = n)
        u_m = np.roll(u, -1, axis = n)
        
        if mode == 'forward':
            du = (u_p - u)/1.0
        elif mode == 'backward':
            du = (u - u_m)/1.0
        else: # default: central difference
            du = (u_p - u_m)/2.0
        grad[n] = du
    pass
    
    return grad

def calculate_tangent_field(grad):
    """
        Calculate the tangent field based on the gradient
    Args:
        grad: the gradient list with length n. Each element in the list 
            corresponds to the partial derivative of that dimension, element 
            size: d1 x d2 ... x dn

    Returns:
        T: the tangent field for each point, size: d1 x d2 x ... x dn x n
    """
    gradnorm = np.sqrt(sum(u**2 for u in grad))
    
    # if the norm is 0, avoid division error by setting it to 1, then set the 
    # all components of the tangent field to 0
    idx_zeros = gradnorm == 0
    gradnorm[idx_zeros] = 1
    T = grad/gradnorm
    for t in T:
        t[idx_zeros] = 0
        
    T = np.dstack(t for t in T)
    pass
    return T

if __name__ == "__main__":
    u = Iout
    grad = calculate_gradient(Iout)
    T = calculate_tangent_field(grad)
    