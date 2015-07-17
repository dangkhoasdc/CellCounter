"""
File: contour.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Contour Class
"""
import numpy as np
import cv2


class Contour(object):
    """Contour Class: The returned result of segmentation method """
    def __init__(self, points_lst):
        """constructor"""
        lefttop, bottomright = Contour.boundary(points_lst)
        self.lefttop = lefttop
        self.bottomright = bottomright

    def draw(self, image, color, thickness=None, ):
        """ draw a rectangle bounding the contour """
        assert type(image) is np.ndarray
        assert type(color) is tuple and len(color) == 3
        if thickness is None:
            thickness = 1
        cv2.rectangle(image, self.topleft, self.bottomright, color, thickness)

    @staticmethod
    def boundary(points_lst):
        """ Find the topleft and the bottomright of a list of points """
        left = tuple(points_lst[points_lst[:, :, 0].argmin()][0])
        right = tuple(points_lst[points_lst[:, :, 0].argmax()][0])
        top = tuple(points_lst[points_lst[:, :, 1].argmin()][0])
        bottom = tuple(points_lst[points_lst[:, :, 1].argmax()][0])
        return [(left, top), (bottom, right)]

    @property
    def center(self):
        """ return the center of the contour """
        return ((self.topleft[0] + self.bottomright[0]) / 2,
                (self.topleft[1] + self.bottomright[1]) / 2)


def findContour(image):
    """ find all contours in an image """
    conts, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments = [Contour(points_lst) for points_lst in conts]
    return segments
