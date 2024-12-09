#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from data_import import read_dataset
from basis import pca, polynomial_basis_expansion
from regression import linear_least_squares
from inference import linear_inference, softmax

def main():
    images, labels = read_dataset('./image_data.npy', './user_ids.npy')
    p = get_pcs(images)
    accuracies = eval_pcs(images, p, labels)
    plt.plot(accuracies)
    plt.show()

def save_pcs(images, path):
    p, cov = pca(images)
    np.save(path, p)
    return p

def get_pcs(images, path='./pcs.npy', force=False):
    if force or not os.path.exists(path):
        pcs = save_pcs(images, path)
    else:
        pcs = np.load(path)
    return pcs

def eval_pcs(images, pcs, labels):
    accuracies = []
    for k in range(1, 100):
        p = pcs[:k, :]
        images_pcs = images @ p.T # To PC basis
        w = linear_least_squares(images_pcs, labels)
        predicted = np.argmax(linear_inference(images_pcs, w) @ p, axis=0)
        accuracies.append(np.mean(predicted == labels))
    return accuracies
        

    

if __name__ == '__main__':
    main()

