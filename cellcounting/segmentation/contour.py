"""
File: contour.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Contour Class
"""
from .. import common as com
import numpy as np
import cv2


class Contour(object):
    """Contour Class: The returned result of segmentation method """
    def __init__(self, points_lst=None):
        """constructor"""
        if points_lst != None:
            lefttop, rightbottom = Contour.boundary(points_lst)
            self.lt = lefttop
            self.rb = rightbottom

    def draw(self, image, color, thickness=None, ):
        """ draw a rectangle bounding the contour """
        assert type(image) is np.ndarray
        assert type(color) is tuple and len(color) == 3
        if thickness is None:
            thickness = 1
        cv2.rectangle(image, tuple(self.rb), tuple(self.lt), color, thickness)

    @staticmethod
    def boundary(points_lst):
        """ Find the lefttop and the rightbottom of a list of points """
        left = tuple(points_lst[points_lst[:, :, 0].argmin()][0])
        right = tuple(points_lst[points_lst[:, :, 0].argmax()][0])
        top = tuple(points_lst[points_lst[:, :, 1].argmin()][0])
        bottom = tuple(points_lst[points_lst[:, :, 1].argmax()][0])
        return [[left[0], top[1]], [right[0], bottom[1]]]

    @property
    def center(self):
        """ return the center of the contour """
        return ((self.lt[0] + self.rb[0]) / 2,
                (self.lt[1] + self.rb[1]) / 2)


    @property
    def width(self):
        """ return the width of the rectangle bounding the contour """
        return abs(self.rb[0] - self.lt[0])

    @property
    def height(self):
        """ return the width of the rectangle bounding the contour """
        return abs(self.rb[1] - self.lt[1])

    def get_region(self, image):
        if type(image) is not np.ndarray:
            raise TypeError("image must be a ndarray")
        return image[self.lt[1]: self.rb[1], self.lt[0]: self.rb[0]]

    def __str__(self):
        return str(self.lt) + ":" + str(self.rb)

    def __eq__(self, other):
        return self.lt[0] == other.lt[0] \
            and self.lt[1] == other.lt[1] \
            and self.rb[0] == other.rb[0] \
            and self.rb[1] == other.rb[1] \

    def __key(self):
        return (self.lt[0], self.lt[1], self.rb[0], self.rb[1])

    def __hash__(self):
        return hash(self.__key())




def findContours(image):
    """ find all contours in an image """
    conts, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(demo, conts, -1, (255, 255, 0), 2)
    # new_conts = []
    # for c in conts:
        # cnt_len = cv2.arcLength(c, False)
        # cnt = cv2.approxPolyDP(c, 0.08 * cnt_len, True)
        # if cv2.contourArea(cnt) > 25: #and cv2.isContourConvex(cnt):
            # new_conts.append(c)

    # conts = new_conts
    segments = [Contour(points_lst) for points_lst in conts]
    return segments
