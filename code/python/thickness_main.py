#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:50:30 2017

@author: fantasie
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import diffusion
import geometry

def main():
    pass

imgfolder = '../../img/'
resultfolder = '../../result/'
imgname = '6.png'

if __name__ == "__main__":
    # TODO: move this to main() after debugging
    img = cv.imread(imgfolder + imgname, cv.IMREAD_COLOR)
    b,g,r = cv.split(img) # get b, g, r
    img = cv.merge([r,g,b]) # switch it to r, g, b
    imshow(img)
    
    boundin = (g > 150) # inner boundary drawn in green
    boundout = (r > 150) # outer boundary drawn in red
    region = (b > 200) # region colored in blue
    inside = (r + g + b < 10) # inside exterior
    outside = (r == 127) # outside exterior
    exterior = np.logical_or(inside, outside)
    bound_points = np.logical_or(boundin, boundout) # boundary points
    
    Iin = np.zeros(b.shape)
    Iin[boundin] = 0.0
    Iin[boundout] = 1.0
    Iin[outside] = 1.0
    Iin[region] = .5
    
    # mask: only focus on the region within the enclosing bounding box
    (xidx, yidx) = (bound_points == True).nonzero()
    xmin, xmax, ymin, ymax = xidx.min(), xidx.max(), yidx.min(), yidx.max()
    mask = np.logical_not(np.zeros(b.shape, dtype = bool))
    mask[xmin:xmax, ymin:ymax] = False
    
    fixed_points = np.logical_not(region)
    
    u = diffusion.linear_heat_diffusion(Iin, fixed_points = fixed_points, maxiter = 4000)
    imshow(u)
    np.save(resultfolder + 'u' + imgname.split('.')[0], u)
    u = np.load(resultfolder + 'u' + imgname.split('.')[0] + '.npy')
    ''' Calculate the tangent field 
    '''
    grad = geometry.calculate_gradient(u)
    ux, uy = grad[0], grad[1]
    T = geometry.calculate_tangent_field(grad)
    Tx, Ty = T[:,:,0], T[:,:,1]
    denom = np.sum(abs(T), axis = 2)
    denom_min = min(denom[region])
    
    ''' Test the update scheme
    '''
    L0 = np.zeros(u.shape)
    L1 = np.zeros(u.shape)
    
    L0old = np.zeros(u.shape)
    L1old = np.zeros(u.shape)
    
    stop = False
    count = 0
    while not stop:
        numer0 = np.ones(u.shape)
        numer1 = np.ones(u.shape)
        for n in xrange(u.ndim):
            L0_p = np.roll(L0old, -1, axis = n)
            L0_m = np.roll(L0old, 1, axis = n)
            L1_p = np.roll(L1old, -1, axis = n)
            L1_m = np.roll(L1old, 1, axis = n)
            numer0 += abs(T[:,:,n]) * \
                      ((T[:,:,n] > 0) * L0_m  + (T[:,:,n] < 0) * L0_p)
            numer1 += abs(T[:,:,n]) * \
                      ((T[:,:,n] > 0) * L1_p  + (T[:,:,n] < 0) * L1_m)
        
        denom[denom == 0] = 1.0
        L0 = numer0/denom
        L1 = numer1/denom
        L0[boundin] = 0.0
        L1[boundout] = 0.0
        L0[exterior] = 0.0
        L1[exterior] = 0.0
        
        count += 1
        if count > 10000:
            break;
            
        L0old, L1old = L0, L1
    
    W = L0 + L1
    W[exterior] = 0
    imshow(W)
        
    
    
    
    
    
    
    
    