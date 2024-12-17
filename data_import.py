#!/usr/bin/env python
import cv2
import os
import numpy as np
from sklearn import preprocessing
from datasets import load_dataset

def preprocess_dataset(path='att_dataset'):
    path = 'att_dataset'
    train_images = []
    test_images = []
    train_user_ids = []
    test_user_ids = []

    user_num = -1
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            user_num += 1
            i = 0
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if i <= 7 and user_num <= 34:
                    train_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    train_images.append(train_image.flatten())
                    train_user_ids.append(user_num)
                else:
                    test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    test_images.append(test_image.flatten())
                    test_user_ids.append(user_num)
                i += 1

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_user_ids = np.array(train_user_ids)
    test_user_ids = np.array(test_user_ids)

    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_user_ids)
    np.save('test_images.npy', test_images)
    np.save('test_labels.npy', test_user_ids)

def binary_face_dataset():
    non_faces = []
    ds = load_dataset("TencentARC/Plot2Code", split="test")

    test_indices = np.random.choice(len(ds), size=120, replace=False)
    all_indices = np.arange(len(ds))
    train_indices = np.setdiff1d(all_indices, test_indices)

    for idx in all_indices:
        idx = int(idx)
        image = ds[idx]['image']
        image_np = np.array(image.convert('L')) # makes grayscale
        resized_image = cv2.resize(image_np, (92, 112))
        non_faces.append(resized_image.flatten())

    non_faces_array = np.array(non_faces)

    non_faces_train = non_faces_array[train_indices]
    non_faces_test = non_faces_array[test_indices]

    faces_train = np.load('train_images.npy')
    faces_test = np.load('test_images.npy')

    binary_train_data = np.concatenate((faces_train, non_faces_train))
    binary_test_data = np.concatenate((faces_test, non_faces_test))

    binary_train_labels = np.concatenate((np.ones(len(faces_train)), np.zeros(len(non_faces_train))))
    binary_test_labels = np.concatenate((np.ones(len(faces_test)), np.zeros(len(non_faces_test))))

    np.save('binary_train_images.npy', binary_train_data)
    np.save('binary_train_labels.npy', binary_train_labels)
    np.save('binary_test_images.npy', binary_test_data)
    np.save('binary_test_labels.npy', binary_test_labels)

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

def read_dataset(images_file, labels_file):
    return np.load(images_file), np.load(labels_file)

if __name__ == '__main__':
    preprocess_dataset()
