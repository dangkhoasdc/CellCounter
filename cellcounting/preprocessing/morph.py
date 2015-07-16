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


def close(img, kernel_dilation, kernel_erosion=None, iters=1):
    """ closing operation """
    assert type(img) is np.ndarray and img.size > 0
    assert type(kernel_dilation) is np.ndarray and kernel_dilation.size > 0

    result = cv2.dilate(img, kernel_dilation, iterations=iters)
    if kernel_erosion is None:
        kernel_erosion = kernel_dilation
    return cv2.erode(result, kernel_erosion, iterations=iters)


def opening(img, kernel_erosion, kernel_dilation=None, iters=1):
    """ opening operation """
    assert type(img) is np.ndarray and img.size > 0
    assert type(kernel_erosion) is np.ndarray and kernel_erosion.size > 0

    result = cv2.erode(img, kernel_erosion, iterations=iters)
    if kernel_erosion is None:
        kernel_dilation = kernel_erosion
    return cv2.dilate(result, kernel_dilation, iterations=iters)


def hitmiss(img, kernel, iters):
    """hit and miss operation """
    yield


def skeletonize(img, kernel, iters):
    """ skeletonize operation """
    yield
