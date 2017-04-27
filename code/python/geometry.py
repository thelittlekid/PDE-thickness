#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:31:40 2017

@author: fantasie
"""

import numpy as np
import numpy.linalg as LA
import heapq as hp
from matplotlib.pyplot import imshow

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
        u_p = np.roll(u, -1, axis = n)
        u_m = np.roll(u, 1, axis = n)
        
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

def iterative_relaxation(T, boundin, boundout, exterior, precision = 1e-6, maxiter = 10000):
    """
        Implementation of Iterative Relaxation algorithm, currently work in 2D
    Reference:
        Yezzi, Anthony J., and Jerry L. Prince. "An Eulerian PDE approach 
        for computing tissue thickness." IEEE transactions on medical 
        imaging 22.10 (2003): 1332-1339.
    Args:
        T: the tangent field of each point, size d1 x d2 x ... x dn x n, where 
            n is the number of dimensions
        boundin, boundout: d1 x d2 x ... x dn binary matrices specifing the 
            inside and outside boundary respectively
        exterior: d1 x d2 x ... dn binary matrix specifying the exterior of the 
            region R
        precision: minimum precision requirement
        maxiter: maximum number of iterations

    Returns:
        W: the thickness of each point, size d1 x d2 x ... x dn
        L0: the distance to the inner boundary, same size as W
        L1: the distance to the outer boundary, same size as W
    """
    # Precalculate the denominator
    denom = np.sum(abs(T), axis = T.ndim - 1) 
    
    # Step 1: set L0 = L1 = 0 at all grid
    L0 = np.zeros(denom.shape)
    L1 = np.zeros(denom.shape)
    
    # Step 2: Use (8) and (9) to update L0 and L1 at points inside R
    # Step 3: repeat step 2 until the values L0 and L1 converges
    L0old = np.zeros(denom.shape)
    L1old = np.zeros(denom.shape)

    stop = False
    count = 0
    while not stop:
        numer0 = np.ones(denom.shape)
        numer1 = np.ones(denom.shape)
        
        # update formula (8) and (9)
        for n in xrange(denom.ndim):
            L0_p = np.roll(L0old, -1, axis = n)
            L0_m = np.roll(L0old, 1, axis = n)
            L1_p = np.roll(L1old, -1, axis = n)
            L1_m = np.roll(L1old, 1, axis = n)
            numer0 += abs(T[:,:,n]) * \
                      ((T[:,:,n] > 0) * L0_m  + (T[:,:,n] < 0) * L0_p)
            numer1 += abs(T[:,:,n]) * \
                      ((T[:,:,n] > 0) * L1_p  + (T[:,:,n] < 0) * L1_m)
        
        denom[denom == 0] = 1.0 # for zero denominators
        # fixed points
        L0, L1 = numer0/denom, numer1/denom
        L0[boundin], L1[boundout] = 0.0, 0.0
        L0[exterior], L1[exterior] = 0.0, 0.0

        count += 1
        if count > maxiter:
            break;
            
        diffnorm0 = LA.norm(L0 - L0old, np.inf)
        diffnorm1 = LA.norm(L1 - L1old, np.inf)
        if(diffnorm0 < precision and diffnorm1 < precision):
            stop = True
            
        L0old, L1old = L0, L1
        
    print "number of iterations in iterative relaxation: ", count
    W = L0 + L1
    W[exterior] = 0
    
    pass
    return W, L0, L1

UNVISITED, VISITED, SOLVED = 0, 1, 2 # status codes
points = []
order = np.zeros([64, 64], dtype='int') - 1
visit_order = np.zeros([64, 64], dtype='int') - 1
def ordered_traversal(shape, T, boundin, boundout, region):
    """
        Implementation of Ordered Traversal algorithm, currently work in 2D
    Reference:
        Yezzi, Anthony J., and Jerry L. Prince. "An Eulerian PDE approach 
        for computing tissue thickness." IEEE transactions on medical 
        imaging 22.10 (2003): 1332-1339.
    Args:
        shape: tuple, specify the shape of the region
        T: the tangent field of each point, size d1 x d2 x ... x dn x n, where 
            n is the number of dimensions
        boundin, boundout: d1 x d2 x ... x dn binary matrices specifing the 
            inside and outside boundary respectively
        region: d1 x d2 x ... dn binary matrix specifying the interior of the 
            region R

    Returns:
        W: the thickness of each point, size d1 x d2 x ... x dn
    """
    count = 1
    L0, L1 = np.zeros(shape), np.zeros(shape)
#    exterior = np.logical_not(np.logical_or(boundin, boundout, region))
#    L0[exterior] = 10000

    # L0[boundin] = 0
#    h = [] # min heap 
    h = {}  # list of VISTIED points, using python dictionary: [position: L]
    
    # Step 1: Initially tag all points in R as UNVISITED    
    Status = np.zeros(shape) + SOLVED # set exterior to be SOLVED
    # Status[boundin] = SOLVED # set initial boundary to be SOLVED
    Status[boundout] = UNVISITED
    Status[region] = UNVISITED # TODO: what about the outer boundary?
    
    # Step 2: Solve for L0 at points next to the inner boundary \partial R0, 
    # and retag these points as VISITED
    # TODO: Will some points be calculated more than once?
    indices = np.dstack(idx for idx in np.where(boundin)) # [1 x #points x dim]
    for i in xrange(indices.shape[1]):
        # get points on the frontier, [0] for dimension issue
        pos = indices[:,i,:][0] # pos is an array [n x ], n is the dimension
        
        # update the neighbors of point pos
        fast_marching(L0, pos, T, Status, h, direction = 0)
    
    # Step 5: Stop if all points in have been tagged SOLVED, else go to Step 3.
    while len(h) > 0: # len(h) > 0
        # Step 3: Find the grid point, within the current list of VISITED points, 
        # with the smallest value of computed so far. Remove this point from the
        # list and re-tag it as SOLVED.
        
#        pos = np.array(hp.heappop(h)[1])
        pos = min(h, key = h.get)
        h.pop(pos)
        
        pos = np.array(pos)
        Status[tuple(pos)] = SOLVED
        points.append(tuple(pos))
        order[tuple(pos)] = count
        count += 1
        
        # Step 4: Update the values of using (8) for whichever neighbors of this
        # grid point are not yet tagged as SOLVED. If any of these neighbors are
        # currently tagged as UNVISITED, re-tag them as VISITED and add them to
        # the current list of VISITED points.
        fast_marching(L0, pos, T, Status, h, direction = 0)
    
    ''' For L1 '''
#    h = [] # min heap
    h = {} # list of VISTIED points, using python dictionary: [position: L]
    
    # Step 1: Initially tag all points in R as UNVISITED    
    Status = np.zeros(shape) + SOLVED # set exterior to be SOLVED
    Status[boundin] = UNVISITED
    Status[region] = UNVISITED # TODO: what about the inner boundary?
    
    # Step 2: Solve for L1 at points next to the inner boundary \partial R1, 
    # and retag these points as VISITED
    # TODO: Will some points be calculated more than once?
    indices = np.dstack(idx for idx in np.where(boundout)) # [1 x #points x dim]
    for i in xrange(indices.shape[1]):
        # get points on the frontier, [0] for dimension issue
        pos = indices[:,i,:][0] # pos is an array [n x ], n is the dimension
        
        # update the neighbors of point pos
        fast_marching(L1, pos, T, Status, h, direction = 1)
    
    # Step 5: Stop if all points in have been tagged SOLVED, else go to Step 3.
    while len(h) > 0:
        # Step 3: Find the grid point, within the current list of VISITED points, 
        # with the smallest value of computed so far. Remove this point from the
        # list and re-tag it as SOLVED.
#        pos = np.array(hp.heappop(h)[1])
        
        pos = min(h, key = h.get)
        h.pop(pos)
        
        pos = np.array(pos)        
        Status[tuple(pos)] = SOLVED
        
        # Step 4: Update the values of using (8) for whichever neighbors of this
        # grid point are not yet tagged as SOLVED. If any of these neighbors are
        # currently tagged as UNVISITED, re-tag them as VISITED and add them to
        # the current list of VISITED points.
        fast_marching(L1, pos, T, Status, h, direction = 1)

    W = L0 + L1
    pass

    return W, L0, L1, Status

def fast_marching(L, pos, T, Status, h, direction = 0):
    """
        Update distance map L for the neighbors around point pos
    Args:
        L: distance map, [d1 x d2 x ... x dn]
        pos: location of the frontier point, [1 x n], n is the dimension
        T: the tangent field of each point, size d1 x d2 x ... x dn x n, where 
            n is the number of dimensions
        Status: status map, [d1 x d2 x ... x dn]
        h: the min heap storing VISITED point
        direction: indicate whether to calculate the distance to inner boundary,
            (L0) as 0, or to calculate the distance to outer boundary (L1) as 1
    Returns:
        L and Status should be updated    
    """
    # Note: python will pass numpy.array by reference
    # pos is a single point, [n x ]
    shape = L.shape
    for dim in xrange(len(pos)): # ndim starts from 0
        shift = np.zeros(len(pos), dtype = 'int')
        for step in [1, -2]: # shift right for 1, then left for 2
            shift[dim] += step
            pos_neighbor = pos + shift # position of the neighbor
            

            if pos_neighbor[dim] < shape[dim] and pos_neighbor[dim] >= 0 \
            and Status[tuple(pos_neighbor)] != SOLVED: # not out of bounds and not SOLVED
                update_value(L, pos_neighbor, T, direction)
                
                # If we use min-heap, we have to update the values of the points
                # that are already in the heap
#                if Status[tuple(pos_neighbor)] == UNVISITED:
#                    hp.heappush(h, (L[tuple(pos_neighbor)], tuple(pos_neighbor)))
                
                # if UNVISITED, add it to the list, if VISITED and already in 
                # the list, update its value. Luckily, same code in python
#                if Status[tuple(pos_neighbor)] == UNVISITED:  
#                    h[tuple(pos_neighbor)] = L[tuple(pos_neighbor)]
                h[tuple(pos_neighbor)] = L[tuple(pos_neighbor)]
                Status[tuple(pos_neighbor)] = VISITED
    pass

def update_value(L, pos, T, direction = 0):
    """
       Calculate the new value for point at pos  
    Args:
        L: distance map, [d1 x d2 x ... x dn]
        pos: location of the frontier point, [1 x n], n is the dimension
        T: the tangent field of each point, size d1 x d2 x ... x dn x n, where 
            n is the number of dimensions
        direction: indicate whether to calculate the distance to inner boundary,
            (L0) as 0, or to calculate the distance to outer boundary (L1) as 1
    Returns:
        L should be updated 
    """
    if tuple(pos) == (53, 23):
        print "Hello"
    shape = L.shape
    numer = 1.0
    denom = sum(abs(T[tuple(pos)]))
    for dim in xrange(len(pos)):
        shift = np.zeros(len(pos), dtype = 'int')
        # find the upwind direction
        if T[tuple(pos)][dim] > 0: 
            shift[dim] += np.sign(direction - 0.5)               
        else:
            shift[dim] -= np.sign(direction - 0.5) 
#        if direction == 0:
#            if T[tuple(pos)][dim] > 0: 
#                shift[dim] -= 1               
#            else:
#                shift[dim] += 1
#        else:
#            if T[tuple(pos)][dim] > 0: 
#                shift[dim] += 1               
#            else:
#                shift[dim] -= 1
            
        pos_neighbor = pos + shift # position of the upwind neighbor
            
        # check if it's not out of bound
        if pos_neighbor[dim] < shape[dim] and pos_neighbor[dim] >= 0:
            value_neighbor = L[tuple(pos_neighbor)]
        else:
            value_neighbor = L[tuple(pos)]
        
            
        numer += abs(T[tuple(pos)][dim]) * value_neighbor
        
    L[tuple(pos)] = numer / denom
    pass

if __name__ == "__main__":
    W1, L0, L1, Status = ordered_traversal(boundin.shape, T, boundin, boundout, region)
    imshow(L0)
    pass
    