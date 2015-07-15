"""
File: common.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Helper functions for Cell Counting program
"""

import scipy.spatial.distance as dist

def euclid(p1, p2):
    """ Calculate the Euclidean distance between two points """
    return dist.euclidean(p1, p2)
