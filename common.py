"""
File: common.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Helper functions for Cell Counting program
"""
import numpy as np
import scipy.spatial.distance as dist
import cv2
from matplotlib import pyplot as plt


def euclid(p1, p2):
    """ Calculate the Euclidean distance between two points """
    return dist.euclidean(p1, p2)


def drawHist(image, channel):
    """Draw the histogram of an input image """
    assert isinstance(image, basestring)
    if channel == 1:
        im = cv2.imread(image, 0)
        plt.hist(im.ravel(), 256, [0,256]); plt.show()
    elif channel == 3:
        im = cv2.imread(image, 1)
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histr = cv2.calcHist([im], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
