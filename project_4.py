#!/usr/bin/env python

import numpy as np
from data_import import read_mnist

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

def linear_least_squares(x, labels):
    y = np.zeros((labels.size, labels.max() + 1))
    y[np.arange(labels.size), labels] = 1
    # x: (num_samples, num_features) y: (num_samples, num_classes)
    x_squared = x.T @ x
    w = y.T @ x @ np.linalg.inv(x_squared)
    return w


def softmax(x):
    e_x = np.exp(x - np.max(x))



def main():
    images, labels = read_mnist()
    # images = images[labels==4] 
    # labels = labels[labels==4]
    p, cov = pca(images)
    p = p[:90, :]
    images_pcs = images @ p.T
    w = linear_least_squares(images_pcs, labels)
    print(w @ (images[labels==4][0] @ p.T))



if __name__ == '__main__':
    main()

