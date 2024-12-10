#!/usr/bin/env python
import numpy as np
from itertools import combinations_with_replacement

def pca(x, method='eigen'):
    # x is `num_points` by `num_features`
    x_centered = x - np.mean(x, axis=0)
    num_points = x.shape[1]
    if method == 'eigen':
        eig, p = np.linalg.eigh( (x_centered.T @ x_centered) / (num_points-1) )
        indices = np.argsort(eig)[::-1]
        cov = np.diag(eig[indices])
        p = p[indices]
    elif method == 'svd':
        u, s, _ = np.linalg.svd( (x_centered.T @ x_centered) / np.sqrt(num_points-1) )
        p = u
        cov = s**2
    else:
        raise ValueError(f"Unknown method provided: {method}.\nSupported methods are 'eigen' and 'svd'.")
    return p, cov

def polynomial_basis_expansion(x, k=0, interactions=True):
    if k < 0:
        raise ValueError(f"Invalid order: {k}.\nThe polynomial order `k` must be non-negative.")
    num_points, num_features = x.shape
    expanded_features = [np.ones(num_points)] if k >= 0 else []
    for degree in range(1, k + 1):
        if interactions:
            for term in combinations_with_replacement(range(num_features), degree):
                expanded_features.append(np.prod(x[:, term], axis=1))
        else:
            for feature_idx in range(num_features):
                expanded_features.append(x[:, feature_idx]**degree)
    return np.column_stack(expanded_features)

