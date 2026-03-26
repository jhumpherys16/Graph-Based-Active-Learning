import gl_graph
import objects_utils

import numpy as np
import graphlearning
import random
from scipy import sparse

def GenerateWMat(X, y, n, label_arr, num_samples=500, kernel="uniform"):
    random.seed(93)
    """ 
    Takes data and creates weight matrix
    
    Parameters:
        X (m x n nparray): The data matrix
        y (nparray): The target vector
        label_arr (list): A list of the labels we mask for
        n (int): The number of neighbors"""
    
    # Mask for the labels you want
    mask = np.isin(y, label_arr)
    X_ = X[mask]
    y_ = y[mask]

    # Argsort data
    arg_indices = np.argsort(y_)
    y_sorted = y_[arg_indices]
    X_sorted = X_[arg_indices]

    X_arr = []
    y_arr = []

    for i in range(len(label_arr)):
        
        # 0, 1, ..., len( number of labels matching label_arr[i] )
        offset = sum([len(y_[y_ == label_arr[k]]) for k in range(i)])
        indices = np.arange(offset, len(y_[y_ == label_arr[i]]) + offset)
        # Randomly sample indices
        ran_indices = np.array(random.sample(list(indices), num_samples))

        # Fancy indexing
        X_sample = X_sorted[ran_indices]
        y_sample = y_sorted[ran_indices]
        
        X_arr.append(X_sample)
        y_arr.append(y_sample)

    # Stack sampled data
    X_stack = np.vstack(X_arr)
    y_stack = np.hstack(y_arr)

    J, D = graphlearning.weightmatrix.knnsearch(X_stack, k=n+1, method="brute", similarity="euclidean")
    # From https://github.com/jwcalder/GraphLearning/blob/master/graphlearning/weightmatrix.py
    # On graphlearning.weightmatrix.knn
    # General function for k-nearest neighbor searching, including efficient 
    # implementations for high dimensional data, and support for saving
    # k-nn data to files automatically, for reuse later.
    # If desired, the user can provide knn_data = (knn_ind, knn_dist), the output of a knnsearch,
    # in order to bypass the knnsearch step, which can be slow for large datasets.
    W = graphlearning.weightmatrix.knn(X_stack, k=n, kernel=kernel, symmetrize=True, knn_data=(J, D))
    W = W.tocsr()
    W.setdiag(0)
    W.eliminate_zeros()

    return X_stack, y_stack, W

def GenEvecs(W, num_samples=500):
    random.seed(93)
    W = W.toarray().copy()

    # Top right corner of weight matrix
    W_corner = W[:num_samples, num_samples:]

    # Create set of boundary edges
    boundary_edges = set()
    for i in range(num_samples):
        for j in np.where(W_corner[i] != 0)[0]:
            boundary_edges.add((i, j+num_samples))

    # Find initial values
    G = gl_graph.graph(W)
    evals_init, evecs_init = G.eigen_decomp(k=3)
    evecs_arr = [evecs_init]
    evals_arr = [evals_init]

    # Flip each boundary edge (sorted for deterministic order)
    for tup in sorted(boundary_edges):
        i = tup[0]
        j = tup[1]
        W[i, j] = -1
        W[j, i] = -1

        # Get new eigenvectors and 0th eigenvalue
        G = gl_graph.graph(W)
        evals, evecs = G.eigen_decomp(k=3)
        evals_arr.append(evals)
        # Transform each eigenvector to have closest signage to its previous state
        evec_corrected = objects_utils.Transform(evecs_arr[-1], evecs)
        evecs_arr.append(evec_corrected)

    return evals_arr, evecs_arr