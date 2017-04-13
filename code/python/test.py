#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 23:36:59 2017

@author: fantasie
"""

import numpy as np
import cv2 as cv

#def func(a, b, c):
#    print a, b, c
#
#func('c' = 5, 'b' = 4, 'a' = 3)

#a = np.zeros([3, 4])
#b = np.ones(a.shape)

#a = np.array([[1, 1], [3, 4]])
#b = [1,2]
#
#c =  (a > 2).ravel().nonzero()
#d = a == 1
#
#a[d] = 5

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import diffusion

I = cv.imread('messi.jpg', 0)
Iout = diffusion.linear_heat_diffusion(I, maxiter = 1000, dt = 0.20)
#Iout = diffusion.geometric_heat_diffusion(I, maxiter = 1000, dt = 0.20)
plt.subplot(121)
plt.imshow(I, cmap = 'gray')
plt.subplot(122)
plt.imshow(Iout, cmap = 'gray')




