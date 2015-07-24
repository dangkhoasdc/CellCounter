"""
File: morphs.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Morphological operations
"""

import cv2
import numpy as np
import pymorph
from scipy import weave


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


def thinningIter(im, iter):
    """ thinning Iterations """
    assert len(im.shape) == 2

    marker = np.zeros(im.shape, dtype=np.uint8)
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):

            p2 = im[i-1, j]
            p3 = im[i-1, j+1]
            p4 = im[i, j+1]
            p5 = im[i+1, j+1]
            p6 = im[i+1, j]
            p7 = im[i+1, j-1]
            p8 = im[i, j-1]
            p9 = im[i-1, j-1]

            A = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                 (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                 (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                 (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)

            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

            m1 = p2 * p4 * p6 if iter == 0 else p2 * p4 * p8
            m2 = p4 * p6 * p8 if iter == 0 else p2 * p6 * p8

            if A == 1 and (2 <= B <= 6) and m1 == 0 and m2 == 0:
                marker[i, j] = 1


    return im & ~marker


def thinning(img, iters=-1):
    """ thinning morphological operation """
    # convert to binary
    assert len(img.shape) == 2

    bin_img = pymorph.binary(img)

    result = pymorph.thin(bin_img, n=iters)

    return pymorph.gray(result)

