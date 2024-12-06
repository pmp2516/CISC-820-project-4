#!/usr/bin/env python

import numpy as np
from data_import import read_mnist
from basis import pca
from regression import linear_least_squares
from inference import linear_inference, softmax

def main():
    images, labels = read_mnist()
    p, cov = pca(images)
    p = p[:20, :]
    images_pcs = images @ p.T
    w = linear_least_squares(images_pcs, labels)
    print(softmax(linear_inference(images_pcs[labels==4][0], w) @ p))



if __name__ == '__main__':
    main()

