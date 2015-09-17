"""
File: common.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Helper functions for Cell Counting program
"""
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.cm as cm
import cv2
from matplotlib import pyplot as plt
import random
from matplotlib import pyplot as plt


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
    if len(image.shape) == 2:
        plt_im = plt.imshow(image, cmap=cm.Greys_r)
    else:
        plt_im = plt.imshow(image)
    plt.show()


def visualize_segments(image, segments, loc_list):
    """
    Visualize segments
    """
    for loc in loc_list:
        cv2.circle(image, loc, 2, (0, 255, 0))
    for seg in segments:
        if seg.detected:
            seg.draw(image, (0, 0, 255), 1)
        else:
            seg.draw(image, (255, 0, 0), 1)
    return image




