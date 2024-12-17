#!/usr/bin/env python
import numpy as np

def linear_least_squares(x, labels):
    if np.unique(labels).size != 2:
        y = np.zeros((labels.size, 40))
        y[np.arange(labels.size), labels] = 1
    else:
        y = labels
    # x: (num_samples, num_features) y: (num_samples, num_classes)
    x_squared = x.T @ x
    try:
        w = y.T @ x @ np.linalg.pinv(x_squared)
    except np.linalg.LinAlgError:
        return None
    return w

