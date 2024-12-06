#!/usr/bin/env python
import numpy as np

def pca(x, method='eigen'):
    # x is `num_samples` by `num_features`
    x_centered = x - np.mean(x, axis=0)
    num_points = x.shape[1]
    if method == 'eigen':
        eig, p = np.linalg.eigh( (x_centered.T @ x_centered) / (num_points-1) )
        indices = np.argsort(eig)
        cov = np.diag(eig[indices])
        p = p[indices]
    elif method == 'svd':
        u, s, _ = np.linalg.svd( (x_centered.T @ x_centered) / np.sqrt(num_points-1) )
        p = u
        cov = s**2
    else:
        raise ValueError(f"Unknown method provided: {method}\nSupported methods are 'eigen' and 'svd'.")
    return p, cov

def basis_expansion(x, k=0):
    pass

