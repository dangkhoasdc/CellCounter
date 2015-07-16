"""
File: thresholding.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Implementations of thresholding algorithms
"""
import numpy as np


def otsu(image, level):
    """ Multilevel otsu thresholding method """
    assert type(image) is np.ndimage
    assert isinstance(level, int)
    yield
