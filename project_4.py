#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from data_import import read_dataset
from basis import pca, polynomial_basis_expansion
from regression import linear_least_squares
from inference import linear_inference, softmax
import cv2 as cv

HEIGHT = 112
WIDTH = 92

def main():
    images, labels = read_dataset('./image_data.npy', './user_ids.npy')
    pcs = get_pcs(images)
    accuracies = eval_pcs(images, pcs, labels, k_vals = np.linspace(10, 1000, num=100, dtype=int))
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

def eval_pcs(images, pcs, labels, k_vals = None):
    if k_vals is None:
        k_vals = np.linspace(10, HEIGHT * WIDTH, num=1000, dtype=int)
    accuracies = []
    print('k,accuracy')
    for k in k_vals:
        p = pcs[:, :k]
        images_pcs = images @ p # To PC basis
        w = linear_least_squares(images_pcs, labels)
        if w is None:
            continue
        output = linear_inference(images_pcs, w)
        predicted = np.argmax(output, axis=1)
        accuracy = np.mean(predicted == labels)
        print(f'{k},{accuracy}')
        accuracies.append(accuracy)
    return accuracies
        
def project_and_reconstruct(images, pcs, k):
    p = pcs[:, :k]
    images_pcs = images @ p
    images_reconstructed = images_pcs @ p.T
    return images_reconstructed

def visualize_reconstructions(images, pcs, k_vals = None):
    if k_vals is None:
        k_vals = np.linspace(1, HEIGHT * WIDTH, num=25, dtype=int)
    num_reconstructions = len(k_vals)
    images_reconstructed_all = np.zeros((400, num_reconstructions + 1, HEIGHT * WIDTH))
    images_reconstructed_all[:, 0] = images
    for i, k in enumerate(k_vals):
        images_reconstructed_all[:, i+1] = project_and_reconstruct(images, pcs, k)

        

    for image_and_reconstructions in images_reconstructed_all:
        fig, axes = plt.subplots(1, num_reconstructions+1, figsize=(20,5))
        for j, ax in enumerate(axes):
            ax.imshow(image_and_reconstructions[j].reshape(HEIGHT, WIDTH), cmap='gray')
            ax.set_title('original' if j == 0 else f"k={k_vals[j-1]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show(block=True)

    

if __name__ == '__main__':
    main()

