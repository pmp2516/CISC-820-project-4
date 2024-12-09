#!/usr/bin/env python
import numpy as np

def softmax(x):
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def linear_inference(x, w):
    return x @ w.T

