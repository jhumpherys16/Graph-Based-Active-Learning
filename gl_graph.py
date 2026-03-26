# From jwcalder
# https://github.com/jwcalder/GraphLearning/tree/master/graphlearning

"""
Graph Class
========================

This module contains the `graph` class, which implements many graph-based algorithms, including
spectral decompositions, distance functions (via Dijkstra and peikonal), PageRank, AMLE (Absolutely 
Minimal Lipschitz Extensions), p-Laplace equations, and basic calculus on graphs.
"""

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg
import sys
import gl_utils
import itertools
import random


class graph:

    def __init__(self, W, labels=None, features=None, label_names=None, node_names=None):
        """Graph class
        ========

        A class for graphs, including routines to compute Laplacians and their
        eigendecompositions, which are useful in graph learning.

        Parameters
        ----------
        W : (n,n) numpy array, matrix, or scipy sparse matrix
            Weight matrix representing the graph.
        labels : (n,) numpy array (optional)
            Node labels.
        features : (n,k) numpy array (optional)
            Node features.
        label_names : list (optional)
            Names corresponding to each label.
        node_names : list (optional)
            Names for each node in the graph.
        """

        self.weight_matrix = sparse.csr_matrix(W)
        self.labels = labels
        self.features = features
        self.num_nodes = W.shape[0]
        self.label_names = label_names
        self.node_names = node_names

        self.__ccode_init__()

        self.eigendata = {}
        normalizations = ['combinatorial','randomwalk','normalized']

        for norm in normalizations:
            self.eigendata[norm] = {}
            self.eigendata[norm]['eigenvectors'] = None
            self.eigendata[norm]['eigenvalues'] = None
            self.eigendata[norm]['method'] = None
            self.eigendata[norm]['k'] = None
            self.eigendata[norm]['c'] = None
            self.eigendata[norm]['gamma'] = None
            self.eigendata[norm]['tol'] = None
            self.eigendata[norm]['q'] = None

    def __ccode_init__(self):

        #Coordinates of sparse matrix for passing to C code
        I,J,V = sparse.find(self.weight_matrix)
        ind = np.argsort(I)
        self.I,self.J,self.V = I[ind], J[ind], V[ind]
        self.K = np.array((self.I[1:] - self.I[:-1]).nonzero()) + 1
        self.K = np.append(0,np.append(self.K,len(self.I)))
        self.Vinv = 1/self.V

        #For passing to C code
        self.I = np.ascontiguousarray(self.I, dtype=np.int32)
        self.J = np.ascontiguousarray(self.J, dtype=np.int32)
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.Vinv = np.ascontiguousarray(self.Vinv, dtype=np.float64)
        self.K = np.ascontiguousarray(self.K, dtype=np.int32)

    
    def eigen_decomp(self, normalization='combinatorial', method='exact', k=10, c=None, gamma=0, tol=0, q=1):
        """Eigen Decomposition of Graph Laplacian
        ======

        Computes the the low-lying eigenvectors and eigenvalues of 
        various normalizations of the graph Laplacian. Computations can 
        be either exact, or use a fast low-rank approximation via 
        randomized SVD. 

        Parameters
        ----------
        normalization : {'combinatorial','randomwalk','normalized'}, default='combinatorial'
            Type of normalization of graph Laplacian to apply.
        method : {'exact','lowrank'}, default='exact'
            Method for computing eigenvectors. 'exact' uses scipy.sparse.linalg.svds, while
            'lowrank' uses a low rank approximation via randomized SVD. Lowrank is not 
            implemented for gamma > 0.
        k : int (optional), default=10
            Number of eigenvectors to compute.
        c : int (optional), default=2*k
            Cutoff for randomized SVD.
        gamma : float (optional), default=0
            Parameter for modularity (add more details)
        tol : float (optional), default=0
            tolerance for eigensolvers.
        q : int (optional), default=1
            Exponent to use in randomized svd.

        Returns
        -------
        vals : numpy array, float 
            eigenvalues in increasing order.
        vecs : (n,k) numpy array, float
            eigenvectors as columns.

        Example
        -------
        This example compares the exact and lowrank (ranomized svd) methods for computing the spectrum: [randomized_svd.py](https://github.com/jwcalder/GraphLearning/blob/master/examples/randomized_svd.py).
        ```py
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn.datasets as datasets
        import graphlearning as gl

        X,L = datasets.make_moons(n_samples=500,noise=0.1)
        W = gl.weightmatrix.knn(X,10)
        G = gl.graph(W)

        num_eig = 7
        vals_exact, vecs_exact = G.eigen_decomp(normalization='normalized', k=num_eig, method='exact')
        vals_rsvd, vecs_rsvd = G.eigen_decomp(normalization='normalized', k=num_eig, method='lowrank', q=50, c=50)

        for i in range(1,num_eig):
            rsvd = vecs_rsvd[:,i]
            exact = vecs_exact[:,i]

            sign = np.sum(rsvd*exact)
            if sign < 0:
                rsvd *= -1

            err = np.max(np.absolute(rsvd - exact))/max(np.max(np.absolute(rsvd)),np.max(np.absolute(exact)))

            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
            fig.suptitle('Eigenvector %d, err=%f'%(i,err))

            ax1.scatter(X[:,0],X[:,1], c=rsvd)
            ax1.set_title('Random SVD')

            ax2.scatter(X[:,0],X[:,1], c=exact)
            ax2.set_title('Exact')

        plt.show()
        ```
        """

        #Default choice for c
        if c is None:
            c = 2*k

        same_method = self.eigendata[normalization]['method'] == method
        same_k = self.eigendata[normalization]['k'] == k
        same_c = self.eigendata[normalization]['c'] == c
        same_gamma = self.eigendata[normalization]['gamma'] == gamma
        same_tol = self.eigendata[normalization]['tol'] == tol
        same_q = self.eigendata[normalization]['q'] == q

        #If already computed, then return eigenvectors
        if same_method and same_k and same_c and same_gamma and same_tol and same_q:
        
            return self.eigendata[normalization]['eigenvalues'], self.eigendata[normalization]['eigenvectors']
        
        #Else, we need to compute the eigenvectors
        else:
            self.eigendata[normalization]['method'] = method 
            self.eigendata[normalization]['k'] = k
            self.eigendata[normalization]['c'] = c
            self.eigendata[normalization]['gamma'] = gamma
            self.eigendata[normalization]['tol'] = tol
            self.eigendata[normalization]['q'] = q

            n = self.num_nodes

            #If not using modularity
            if gamma == 0:
                
                if normalization == 'randomwalk' or normalization == 'normalized':

                    D = self.degree_matrix(p=-0.5)
                    A = D*self.weight_matrix*D

                    if method == 'exact':
                        u,s,vt = splinalg.svds(A, k=k, tol=tol)
                    elif method == 'lowrank':
                        u,s,vt = gl_utils.randomized_svd(A, k=k, c=c, q=q)
                    else:
                        sys.exit('Invalid eigensolver method '+method)

                    vals = 1 - s
                    ind = np.argsort(vals)
                    vals = vals[ind]
                    vecs = u[:,ind]

                    if normalization == 'randomwalk':
                        vecs = D@vecs

                elif normalization == 'combinatorial':

                    
                    #ORIG
                    """
                    L = self.laplacian()
                    deg = self.degree_vector()
                    M = 2*np.max(deg)
                    A = M*sparse.identity(n) - L

                    if method == 'exact':
                        u,s,vt = splinalg.svds(A, k=k, tol=tol)
                    elif method == 'lowrank':
                        u,s,vt = gl_utils.randomized_svd(A, k=k, c=c, q=q)
                    else:
                        sys.exit('Invalid eigensolver method '+method)
                    
                    vals = M - s
                    ind = np.argsort(vals)
                    vals = vals[ind]
                    vecs = u[:,ind]
                    """
                    
                    
                    L = self.laplacian()
                    if method == 'exact':
                        vals, vecs = splinalg.eigsh(L, k=k, which='SM', tol=tol)
                    else:
                        sys.exit('Lowrank not implemented for combinatorial Laplacian')
                    
                    # After this, the first column from eigen_decomp(..., normalization='combinatorial') is guaranteed to be the smallest-eigenvalue eigenvector, i.e., your intended evec₀.
                    # 🔧 Ensure evec0 truly corresponds to the smallest eigenvalue
                    ind = np.argsort(vals)
                    vals = vals[ind]
                    vecs = vecs[:, ind]

                else:
                    sys.exit('Invalid choice of normalization')


            #Modularity
            else:

                if method == 'lowrank':
                    sys.exit('Low rank not implemented for modularity')

                if normalization == 'randomwalk':
                    lap = self.laplacian(normalization='normalized')
                    P = self.degree_matrix(p=-0.5)
                    p1,p2 = 1.5,0.5
                else:
                    lap = self.laplacian(normalization=normalization)
                    P = sparse.identity(n)
                    p1,p2 = 1,1

                #If using modularity
                deg = self.degree_vector()
                deg1 = deg**p1
                deg2 = deg**p2
                m = np.sum(deg)/2 
                def M(v):
                    v = v.flatten()
                    return (lap*v).flatten() + (gamma/m)*(deg2.T@v)*deg1

                L = sparse.linalg.LinearOperator((n,n), matvec=M)
                vals, vecs = sparse.linalg.eigsh(L, k=k, which='SM', tol=tol)

                #Correct for random walk Laplacian if chosen
                vecs = P@vecs


            #Store eigenvectors for resuse later
            self.eigendata[normalization]['eigenvalues'] = vals
            self.eigendata[normalization]['eigenvectors'] = vecs

            return vals, vecs
        
    def degree_matrix(self, p=1):
            """Degree Matrix
            ======

            Given a weight matrix \\(W\\), returns the diagonal degree matrix 
            in the form
            \\[D_{ii} = \\left(\\sum_{j=1}^n w_{ij}\\right)^p.\\]

            Parameters
            ----------
            p : float (optional), default=1
                Optional exponent to apply to the degree.

            Returns
            -------
            D : (n,n) scipy sparse matrix, float
                Sparse diagonal degree matrix.
            """

            # Absolute value for signed laplacian
            diags = np.array(np.abs(self.weight_matrix).sum(axis=1))
            diags = list(itertools.chain.from_iterable(diags))
            D = sparse.diags(diags)
            return D
    



    def laplacian(self, normalization="combinatorial", alpha=1):
        """Graph Laplacian
        ======

        Computes various normalizations of the graph Laplacian for a 
        given weight matrix \\(W\\). The choices are
        \\[L_{\\rm combinatorial} = D - W,\\]
        \\[L_{\\rm randomwalk} = I - D^{-1}W,\\]
        and
        \\[L_{\\rm normalized} = I - D^{-1/2}WD^{-1/2},\\]
        where \\(D\\) is the diagonal degree matrix, which is defined as
        \\[D_{ii} = \\sum_{j=1}^n w_{ij}.\\]
        The Coifman-Lafon Laplacian is also supported. 

        Parameters
        ----------
        normalization : {'combinatorial','randomwalk','normalized','coifmanlafon'}, default='combinatorial'
            Type of normalization to apply.
        alpha : float (optional)
            Parameter for Coifman-Lafon Laplacian

        Returns
        -------
        L : (n,n) scipy sparse matrix, float
            Graph Laplacian as sparse scipy matrix.
        """

        I = sparse.identity(self.num_nodes)
        D = self.degree_matrix()

        if normalization == "combinatorial":
            L = D - self.weight_matrix
        elif normalization == "randomwalk":
            Dinv = self.degree_matrix(p=-1)
            L = I - Dinv*self.weight_matrix
        elif normalization == "normalized":
            Dinv2 = self.degree_matrix(p=-0.5)
            L = I - Dinv2*self.weight_matrix*Dinv2
        elif normalization == "coifmanlafon":
            D = self.degree_matrix(p=-alpha)
            L = graph(D*self.weight_matrix*D).laplacian(normalization='randomwalk')
        else:
            sys.exit("Invalid option for graph Laplacian normalization.")

        return L.tocsr()
    

    def degree_vector(self):
        """Degree Vector
        ======

        Given a weight matrix \\(W\\), returns the diagonal degree vector
        \\[d_{i} = \\sum_{j=1}^n w_{ij}.\\]

        Returns
        -------
        d : numpy array, float
            Degree vector for weight matrix.
        """

        d = np.asarray(np.abs(self.weight_matrix).sum(axis=1)).ravel()
        return d