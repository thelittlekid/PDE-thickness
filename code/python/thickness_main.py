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

def main():
    pass

imgfolder = '../../img/'
resultfolder = '../../result/'
imgname = '7.png'

if __name__ == "__main__":
    # TODO: move this to main() after debugging
    img = cv.imread(imgfolder + imgname, cv.IMREAD_COLOR)
    b,g,r = cv.split(img) # get b, g, r
    img = cv.merge([r,g,b]) # switch it to r, g, b
    imshow(img)
    
    boundin = (g > 150) # inner boundary drawn in green
    boundout = (r > 150) # outer boundary drawn in red
    region = (b > 200) # region colored in blue
    bound_points = np.logical_or(boundin, boundout) # boundary points
    
    Iin = np.zeros(b.shape)
    Iin[boundin] = 0.0
    Iin[boundout] = 1.0
    Iin[region] = .5
    
    # mask: only focus on the region within the enclosing bounding box
    (xidx, yidx) = (bound_points == True).nonzero()
    xmin, xmax, ymin, ymax = xidx.min(), xidx.max(), yidx.min(), yidx.max()
    mask = np.logical_not(np.zeros(b.shape, dtype = bool))
    mask[xmin:xmax, ymin:ymax] = False
    
    fixed_points = np.logical_not(region)
    
    Iout = diffusion.linear_heat_diffusion(Iin, fixed_points = fixed_points, maxiter = 1000)
    imshow(Iout)
    np.save(resultfolder + 'u' + imgname.split('.')[0], Iout)
    
    
    