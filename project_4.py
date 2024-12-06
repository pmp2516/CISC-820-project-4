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

def linear_least_squares(x, labels):
    y = np.zeros((labels.size, labels.max() + 1))
    y[np.arange(labels.size), labels] = 1
    # x: (num_samples, num_features) y: (num_samples, num_classes)
    x_squared = x.T @ x
    w = y.T @ x @ np.linalg.inv(x_squared)
    return w

def read_mnist():
    """
    Reads MNIST handwritten digit training data and labels.

    Returns:
        image: A 2D NumPy array of shape (28*28, 60000), each column represents an image.
        label: A 1D NumPy array of length 60000, each element represents a label.
    """
    with open('train-images-idx3-ubyte', 'rb') as fid, open('train-labels-idx1-ubyte', 'rb') as fid2:
        # Read headers (first 4 bytes for magic number, next 4 for dimensions)
        mn = np.frombuffer(fid.read(4), dtype=np.uint8)
        ni = np.frombuffer(fid.read(4), dtype=np.uint8)
        nr = np.frombuffer(fid.read(4), dtype=np.uint8)
        nc = np.frombuffer(fid.read(4), dtype=np.uint8)
        
        mn2 = np.frombuffer(fid2.read(4), dtype=np.uint8)
        ni2 = np.frombuffer(fid2.read(4), dtype=np.uint8)

        # Read labels (60000 bytes)
        label = np.frombuffer(fid2.read(60000), dtype=np.uint8)
        
        # Read images (60000 images, each 28*28 bytes)
        image = np.zeros((60000, 28 * 28), dtype=np.uint8)
        for i in range(60000):
            image[i, :] = np.frombuffer(fid.read(28 * 28), dtype=np.uint8)
        
    return image, label

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

