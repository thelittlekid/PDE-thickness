#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 23:29:23 2017

@author: fantasie
"""
import numpy as np
import math
import numpy.linalg as LA

def linear_heat_diffusion(Iin, dt = 0.2, fixed_points = [], maxiter = 100):
    """Diffuses a grayscale image with linear heat equation

    Args:
        Iin: the input image in grayscale
        dt: the time step, set default 0.25 to satisfy the CFL condition
            when dx and dy are both 1
        fixed_points: boolean matrix specifying fixed points which shall NOT be
            smoothed
        maxiter: maximum number of iterations

    Returns:
        The diffused version of the input grayscale image
    """
    Iold = np.array(Iin, dtype=float)
    I = np.zeros(Iin.shape)
    
    stop = False
    
    if dt > .25:
        print 'CFL condition not satisfied, dt must be less than ', .25 * min(dx, dy)**2 
#        return Iin
    
    count = 0
    while (not stop): 
        # the shifted matrices
        I_px = np.roll(Iold, -1, axis = 0)
        I_mx = np.roll(Iold, 1, axis = 0)
        I_py = np.roll(Iold, -1, axis = 1)
        I_my = np.roll(Iold, 1, axis = 1)
        # pad the boundary with the nearest neighbor
        I_px[0,] = Iold[0,]
        I_mx[I_mx.shape[0] - 1,] = Iold[Iold.shape[0] - 1,]
        I_py[:, 0] = Iold[:, 0]
        I_my[:, I_my.shape[1] - 1] = Iold[:, Iold.shape[1] - 1]
        
        # update formula
        I = Iold + dt * (I_px + I_mx + I_py + I_my - 4 * Iold)
        I[fixed_points] = Iold[fixed_points]
        
        count += 1
        if count > maxiter:
            break
        
        diffnorm = LA.norm(I - Iold, np.inf)
        if(diffnorm < 1e-6):
            stop = True
            
        Iold = I
    print "number of iteration: ", count
    print "norm of difference: ", diffnorm
    pass
    return I

def geometric_heat_diffusion(Iin, dt = 0.25, fixed_points = [], maxiter = 100):
    """Diffuses a grayscale image with geometric heat equation

    Args:
        Iin: the input image in grayscale
        dt: the time step, set default 0.25 to satisfy the CFL condition
            when dx and dy are both 1
        fixed_points: boolean matrix specifying fixed points which shall NOT be
            smoothed
        maxiter: maximum number of iterations

    Returns:
        The diffused version of the input grayscale image
    """
    Iold = np.array(Iin, dtype=float)
    I = np.zeros(Iin.shape)
    dx, dy = 1, 1 # space step
    
    stop = False
    
    if dt > .25:
        print 'CFL condition not satisfied, dt must be less than ', .25 * min(dx, dy)**2 
#        return Iin
    
    count = 0
    while (not stop): 
        # the shifted matrices
        I_px = np.roll(Iold, -dx, axis = 0)
        I_mx = np.roll(Iold, dx, axis = 0)
        I_py = np.roll(Iold, -dy, axis = 1)
        I_my = np.roll(Iold, dy, axis = 1)
        
        # replicate the boundary term with the closest
        I_px[0,] = Iold[0,]
        I_mx[I_mx.shape[0] - 1,] = Iold[Iold.shape[0] - 1,]
        I_py[:, 0] = Iold[:, 0]
        I_my[:, I_my.shape[1] - 1] = Iold[:, Iold.shape[1] - 1]
        
        I_pxpy = np.roll(I_px, -dy, axis = 1)
        I_pxmy = np.roll(I_px, dy, axis = 1)
        I_mxpy = np.roll(I_mx, -dy, axis = 1)
        I_mxmy = np.roll(I_mx, dy, axis = 1)
        
        I_pxpy[:, 0] = I_px[:, 0]
        I_pxmy[:, I_pxmy.shape[1] - 1] = I_px[:, I_px.shape[1] - 1]
        I_mxpy[:, 0] = I_mx[:, 0]
        I_mxmy[:, I_mxmy.shape[1] - 1] = I_mx[:, I_mx.shape[1] - 1]
        
        
        # update formula
        Ix = (I_px - I_mx)/ (2*dx)
        Iy = (I_py - I_my)/ (2*dy)
        Ixy = (I_pxpy - I_mxpy - I_pxmy + I_mxmy) / (4*dx*dy)
        Ixx = (I_px - 2*Iold + I_mx)/(dx**2)
        Iyy = (I_py - 2*Iold + I_my)/(dy**2)
        
        # take care of the zero denominator: set It to 0 for that point
        denom = Ix**2 + Iy**2
        idx_zeros = denom == 0
        denom[idx_zeros] = 1
        It = (Iy**2*Ixx - 2*Ix*Iy*Ixy + Ix**2*Iyy)/denom
        It[idx_zeros] = 0
        I = Iold + dt * It
        
        I[fixed_points] = Iold[fixed_points]
        
        count += 1
        if count > maxiter:
            break
        
        if(np.array_equal(I, Iold)):
            stop = True
            
        Iold = I
    print "number of iteration: ", count
    pass
    return I
    