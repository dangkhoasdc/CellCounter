"""
File: pooling.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Pooling functions
"""
import numpy as np


def max_pooling(X):
    """
    The max pooling function
    """
    return np.max(X, axis=0)
