"""
File: nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Find contours in an image and count how many cells in
this image without using machine learning technique
"""
from .. import common as com
from ..db import allidb
import cv2
import numpy as np
from .absframework import AbsFramework
from sklearn.preprocessing import normalize

class NoLearningFramework(object):
    """NoLearning Framework class """
    def __init__(self, scale_ratio, preprocessing, segmentation):
        self.scale_ratio = scale_ratio
        self._segmentation = segmentation
        self._preprocess = preprocessing

    def imread(self, fname, flags=1):
        """ load an image and scale it"""
        im = cv2.imread(fname, flags)
        height, width = im.shape[:2]

        height = int(self.scale_ratio * height)
        width  = int(self.scale_ratio * width)

        if im is None:
            raise IOError("Could not load an image ", fname)
        im = cv2.resize(im, (width, height))
        return im

    def run(self, image, loc_list):
        """ run this framework """
        if not isinstance(image, basestring):
            raise TypeError("The parameter image must be instance of basestring ")
        demo_img = self.imread(image, 1)
        processed_img = self.preprocess(demo_img)
        segments = self.segment(processed_img, demo_img)

        # if there is no segment in an image
        if len(segments) == 0:
            print "There is 0 cell in this image "
            return 0
        else:
        # if there are more than 1 segment in this image
        # visualize cells
            for seg in segments:
                seg.draw(demo_img, (255, 255, 0), 1)

            for loc in loc_list:
                cv2.circle(demo_img, loc, 2, (0, 255, 255), -1)

            com.debug_im(demo_img)
            return len(segments)


    def preprocess(self, image):
        """ pre-process an image """
        return self._preprocess.run(image)

    def segment(self, image, demo=None):
        """ segment an image """
        return self._segmentation.run(image, demo)