#!/usr/bin/env python
import numpy as np

def linear_least_squares(x, labels):
    y = np.zeros((labels.size, labels.max() + 1))
    y[np.arange(labels.size), labels] = 1
    # x: (num_samples, num_features) y: (num_samples, num_classes)
    x_squared = x.T @ x
    w = y.T @ x @ np.linalg.inv(x_squared)
    return w
