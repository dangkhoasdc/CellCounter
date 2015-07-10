"""
File: morphs.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Morphological operations
"""

import cv2
import numpy as np


def dilate(img, kernel, iters=1):
    """ dilation operation """
    assert type(img) is np.ndarray and img.size > 0
    assert type(kernel) is np.ndarray and kernel.size > 0

    return cv2.dilate(img, kernel, iterations=iters)


def erode(img, kernel, iters=1):
    """ erosion operation """
    assert type(img) is np.ndarray and img.size > 0
    assert type(kernel) is np.ndarray and kernel.size > 0

    return cv2.erode(img, kernel, iterations=iters)


def close(img, kernel, iters=1):
    """ closing operation """
    assert type(img) is np.ndarray and img.size > 0
    assert type(kernel) is np.ndarray and kernel.size > 0

    result = cv2.dilate(img, kernel, iterations=iters)

    return cv2.erode(result, kernel, iterations=iters)


def opening(img, kernel, iters=1):
    """ opening operation """
    assert type(img) is np.ndarray and img.size > 0
    assert type(kernel) is np.ndarray and kernel.size > 0

    result = cv2.erode(img, kernel, iterations=iters)

    return cv2.dilate(result, kernel, iterations=iters)


def hitmiss(img, kernel, iters):
    """hit and miss operation """
    yield


def skeletonize(img, kernel, iters):
    """ skeletonize operation """
    yield
