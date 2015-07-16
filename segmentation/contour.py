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
    def __init__(self, topleft, rightdown):
        """constructor"""
        self.topleft = topleft
        self.topleft = rightdown

    def draw(self, image, color, thickness=None, ):
        """ draw a rectangle bounding the contour """
        assert type(image) is np.ndarray
        assert type(color) is tuple and len(color) == 3
        if thickness is None:
            thickness = 1
        cv2.rectangle(image, self.topleft, self.rightdown, color, thickness)

    @property
    def center(self):
        """ return the center of the contour """
        return ((self.topleft[0] + self.rightdown[0]) / 2,
                (self.topleft[1] + self.rightdown[1]) / 2)
