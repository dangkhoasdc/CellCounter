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
import random


def euclid(p1, p2):
    """ Calculate the Euclidean distance between two points """
    return dist.euclidean(p1, p2)


def drawHist(image, channel):
    """Draw the histogram of an input image """
    flags = 0 if channel == 1 else 1

    if isinstance(image, basestring):
        im = cv2.imread(image, flags)
    elif type(image) is np.ndarray:
        im = image
    else:
        raise TypeError("The image must be basestring or ndarray")

    if channel == 1:
        plt.hist(im.ravel(), 256, [0, 256]); plt.show()
    elif channel == 3:
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histr = cv2.calcHist([im], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()


def flatten(lst):
    """ flatten a list of lists """
    return sum(lst, [])


def nearest_point(value, point_lst):
    """ Find the nearest point in points list """
    dists = sorted(point_lst, key=lambda x, p=value: np.abs(euclid(x, p)))
    return dists[0], np.abs(euclid(dists[0], value))


def debug_im(image, wait=False):
    """ debug an image """
    code = random.random()
    cv2.imshow("Debug " + str(code), image)
    cv2.waitKey(0)
    if wait:
        cv2.destroyAllWindows()
