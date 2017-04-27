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
imgname = 'test.png'

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
    Iin[boundout], Iin[outside], Iin[region] = 1.0, 1.0, 0.5

    
    # mask: only focus on the region within the enclosing bounding box
#    (xidx, yidx) = (bound_points == True).nonzero()
#    xmin, xmax, ymin, ymax = xidx.min(), xidx.max(), yidx.min(), yidx.max()
#    mask = np.logical_not(np.zeros(b.shape, dtype = bool))
#    mask[xmin:xmax, ymin:ymax] = False
    
    fixed_points = np.logical_not(region)
    
    ''' Solve Laplace Equation '''
    u = diffusion.linear_heat_diffusion(Iin, fixed_points = fixed_points, \
                                        precision = 1e-8, maxiter = 4000)
    imshow(u)
    np.save(resultfolder + 'u' + imgname.split('.')[0], u)
    u = np.load(resultfolder + 'u' + imgname.split('.')[0] + '.npy')
    
    ''' Calculate the tangent field '''
    grad = geometry.calculate_gradient(u)
    ux, uy = grad[0], grad[1]
    T = geometry.calculate_tangent_field(grad)
    Tx, Ty = T[:,:,0], T[:,:,1]
    denom = np.sum(abs(T), axis = 2)
    # denom_min = min(denom[region]) # should be positive
    
    ''' Iterative Relaxation '''
    W, L0, L1 = geometry.iterative_relaxation(T, boundin, boundout, exterior, maxiter = 10000)
    fig = imshow(W)
    cv.imwrite(imgfolder + 'result_' + imgname, W)
    np.save(resultfolder + 'W' + imgname.split('.')[0], W)
    np.save(resultfolder + 'L0' + imgname.split('.')[0], L0)
    np.save(resultfolder + 'L1' + imgname.split('.')[0], L1)
    
#    ''' Ordered Transversal '''
#    W1, L0, L1, Status = geometry.ordered_traversal(boundin.shape, T, boundin, boundout, region)
#    plt.subplot(122)
#    imshow(W1)
        
    
    
    
    
    
    
    
    