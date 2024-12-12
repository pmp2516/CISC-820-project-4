#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from data_import import read_dataset, preprocess_dataset
from basis import pca, polynomial_basis_expansion
from regression import linear_least_squares
from inference import linear_inference, softmax
import cv2 as cv


def main():
    train_images, train_labels = read_dataset('./train_images.npy', './train_labels.npy')
    test_images, test_labels = read_dataset('./test_images.npy', './test_labels.npy')
    all_images = np.concatenate((train_images, test_images), axis=0)
    pcs = get_pcs(all_images)
    # visualize_reconstructions(all_images, pcs)
    w, k = get_weights(train_images, pcs, train_labels)
    eval(w, k, pcs, test_images, test_labels)


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


def get_weights(images, pcs, labels):
    ws = []
    ks = []
    accuracies = []
    for k in range(10, 1000, 10):
        p = pcs[:, :k]
        images_pcs = images @ p  # To PC basis
        w = linear_least_squares(images_pcs, labels)
        if w is None:
            continue
        output = linear_inference(images_pcs, w)
        predicted = np.argmax(output, axis=1)
        accuracy = np.mean(predicted == labels)
        accuracies.append(accuracy)
        ws.append(w)
        ks.append(k)
    best_idx = np.argmax(accuracies)
    best_w = ws[best_idx]
    best_k = ks[best_idx]
    return best_w, best_k


def eval(w, k, pcs, images, labels):
    p = pcs[:, :k]
    images_pcs = images @ p  # To PC basis
    print(f'w: {w.shape}, p: {p.shape}, images_pcs: {images_pcs.shape}, images: {images.shape}')
    output = linear_inference(images_pcs, w)
    predicted = np.argmax(output, axis=1)
    accuracy = np.mean(predicted == labels)
    print(f'k: {k:50} accuracy: {accuracy:10}')


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
        images_reconstructed_all[:, i + 1] = project_and_reconstruct(images, pcs, k)

    for image_and_reconstructions in images_reconstructed_all:
        fig, axes = plt.subplots(1, num_reconstructions + 1, figsize=(20, 5))
        for j, ax in enumerate(axes):
            ax.imshow(image_and_reconstructions[j].reshape(height, width), cmap='gray')
            ax.set_title('original' if j == 0 else f"k={k_vals[j - 1]}")
            ax.axis('off')
        plt.tight_layout()
        plt.show(block=True)


if __name__ == '__main__':
    main()
