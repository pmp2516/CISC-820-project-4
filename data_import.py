#!/usr/bin/env python
import cv2
import os
import numpy as np

def preprocess_dataset(path='att_dataset'):
    path = 'att_dataset'
    images = []
    user_ids = []
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                images.append(image.flatten())
                user_ids.append(subfolder)

    images = np.array(images)
    user_ids = np.array(user_ids)

    return images, user_ids

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

if __name__ == '__main__':
    images, user_ids = preprocess_dataset()
    np.save('image_data.npy', images)
    np.save('user_ids.npy', user_ids)
