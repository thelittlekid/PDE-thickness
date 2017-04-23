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

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import diffusion
#
#I = cv.imread('messi.jpg', 0)
#Iout = diffusion.linear_heat_diffusion(I, maxiter = 1000, dt = 0.20)
##Iout = diffusion.geometric_heat_diffusion(I, maxiter = 1000, dt = 0.20)
#plt.subplot(121)
#plt.imshow(I, cmap = 'gray')
#plt.subplot(122)
#plt.imshow(Iout, cmap = 'gray')

#def add_one(a):
#    a += 1
#
#a = np.zeros((2,2))
#b = 3
#add_one(a[1,1])

import heapq as hp
from heapq import heappush

h = []
A = np.array([1,2,3])
B = np.array([3,4,5])
hp.heappush(h, (0, A))
hp.heappush(h, (0, B))
hp.heappush(h, (-1, A))

#h = []
#heappush(h, (5, (1,2)))
#heappush(h, (7, (1,2)))
#heappush(h, (1, (1,2)))
#heappush(h, (5, (1,2)))







