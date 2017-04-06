#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:56:26 2017

@author: fantasie
"""
#import cv2 as cv

#WINDOW_NAME = "win"
#image = cv.imread("messi.jpg", 0)
#cv.namedWindow(WINDOW_NAME, cv.CV_WINDOW_AUTOSIZE)
#
#cv.startWindowThread()
#
#cv.imshow(WINDOW_NAME, image)
#cv.waitKey()
#cv.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()