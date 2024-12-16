#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from data_import import read_dataset
from basis import pca, polynomial_basis_expansion
from regression import linear_least_squares
from inference import linear_inference, softmax
import cv2 as cv

def main():
    images, labels = read_dataset('./image_data.npy', './user_ids.npy')
    pcs = get_pcs(images)
    eval_pcs(images, pcs, labels)

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
    for k in range(10, 1000, 10):
        p = pcs[:, :k]
        images_pcs = images @ p # To PC basis
        images_expanded = polynomial_basis_expansion(images_pcs, k=5, interactions=False)
        w = linear_least_squares(images_expanded, labels)
        if w is None:
            continue
        # print(f'w: {w.shape}, p: {p.shape}, images_pcs: {images_pcs.shape}, images: {images.shape}')
        output = linear_inference(images_expanded, w)
        predicted = np.argmax(output, axis=1)
        accuracy = np.mean(predicted == labels)
        print(f'k: {k:50} accuracy: {accuracy:10}')
        accuracies.append(accuracy)
    return accuracies
        
def project_and_reconstruct(images, pcs, k):
    p = pcs[:, :k]
    images_pcs = images @ p
    images_reconstructed = images_pcs @ p.T
    return images_reconstructed

def visualize_reconstructions(images, pcs):
    num_reconstructions = 25
    height, width = 112, 92
    images_reconstructed_all = np.zeros((400, num_reconstructions + 1, height * width))
    images_reconstructed_all[:, 0] = images
    # k_vals = np.linspace(10304, 0, num=num_reconstructions, dtype=int)
    k_vals = np.logspace(0, np.log10(height * width), num=num_reconstructions, dtype=int)
    for i, k in enumerate(k_vals):
        images_reconstructed_all[:, i+1] = project_and_reconstruct(images, pcs, k)

        

    for image_and_reconstructions in images_reconstructed_all:
        fig, axes = plt.subplots(1, num_reconstructions+1, figsize=(20,5))
        for j, ax in enumerate(axes):
            ax.imshow(image_and_reconstructions[j].reshape(height, width), cmap='gray')
            ax.set_title('original' if j == 0 else f"k={k_vals[j-1]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show(block=True)


if __name__ == '__main__':
    main()

