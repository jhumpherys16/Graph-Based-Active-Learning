# From jwcalder
# https://github.com/jwcalder/GraphLearning/tree/master/graphlearning


"""
Utilities
==========

This module implements several useful functions that are used throughout the package.
"""

import numpy as np
from scipy import linalg


def randomized_svd(A, k=10, c=None, q=1):
    """Randomized SVD
    ======

    Approximates top k singular values and vectors of A with a randomized
    SVD algorithm.

   
    Parameters
    ----------
    A : numpy array or matrix, scipy sparse matrix, or sparse linear operator
        Matrix to compute SVD of.
    k : int (optional), default=10
        Number of eigenvectors to compute.
    q : int (optional), default=1
        Exponent to use in randomized svd.
    c : int (optional), default=2*k
        Cutoff for randomized SVD.

    Returns
    -------
    u : (n,k) numpy array, float 
        Unitary matrix having left singular vectors as columns. 
    s : numpy array, float
        The singular values.
    vt : (k,n) numpy array, float
        Unitary matrix having right singular vectors as rows.

    Reference
    ---------
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. [Finding structure 
    with randomness: Probabilistic algorithms for constructing approximate matrix 
    decompositions.](https://arxiv.org/abs/0909.4061) SIAM review 53.2 (2011): 217-288.
    """


    if c is None:
        c = 2*k

    n = A.shape[1]

    #Random Gaussian projection
    Omega = np.random.randn(n,c)
    Y = A@Omega
    for i in range(q):
        Y = A@(A.T@Y)

    #QR Factorization
    Q,R = np.linalg.qr(Y)

    #SVD
    B = Q.T@A
    u,s,vt = linalg.svd(B, full_matrices=False)
    u = Q@u

    #Sort singular values from largest to smallest
    ind = np.argsort(-s)
    u = u[:,ind]
    s = s[ind]
    vt = vt[ind,:]

    #Truncate to k
    u = u[:,:k]
    s = s[:k]
    vt = vt[:k,:]

    return u,s,vt
    

