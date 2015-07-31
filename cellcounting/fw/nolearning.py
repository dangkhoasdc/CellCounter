"""
File: nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Find contours in an image and count how many cells in
this image without using machine learning technique
"""
from .. import common as com
import cv2
from .absframework import AbsFramework


class NoLearningFramework(object):
    """NoLearning Framework class """
    def __init__(self, database, preprocessing, segmentation):
        self._segmentation = segmentation
        self._preprocess = preprocessing
        self._db = database

    def imread(self, fname, flags=1):
        """ load an image and scale it"""

        im = cv2.imread(fname, flags)
        if im is None:
            raise IOError("Could not load an image ", fname)
        im = cv2.resize(im, (self._db.size[0], self._db.size[1]))
        return im

    def run(self, image, loc_list, visualize=False):
        """ run this framework """

        if not isinstance(image, basestring):
            raise TypeError("The parameter image must be instance of basestring ")

        demo_img = self.imread(image, 1)
        processed_img, gray_img = self.preprocess(demo_img)
        segments = self.segment(processed_img, gray_img, demo_img)

        correct = 0
        expected_nums = len(loc_list)
        # draw all counted objects in the image
        if visualize:
            for seg in segments:
                seg.draw(demo_img, (0, 255, 0), 1)

        # if there are more than 1 segment in this image
        if not loc_list:
            # visualize true cells
            for loc in loc_list:
                cv2.circle(demo_img, loc, 2, (0, 255, 255), -1)
            # check if each segment is close to one true cell
            for seg in segments:
                if len(loc_list) == 0:
                    break

                point, value = com.nearest_point(seg.center, loc_list)

                if value <= self._db.tol:
                    loc_list.remove(point)
                    correct += 1
                    if visualize:
                        seg.draw(demo_img, (255, 255, 0), 1)

        if visualize:
            print "The number of expected cells: ", expected_nums
            print "The number of cells counting by the program:", len(segments)
            print "The number of true counting cells: ", correct

            com.debug_im(gray_img)
            com.debug_im(processed_img)
            com.debug_im(demo_img, True)

        return correct, len(segments)

    def preprocess(self, image):
        """ pre-process an image """
        return self._preprocess.run(image)

    def segment(self, image, raw_image, demo):
        """ segment an image """
        return self._segmentation.run(image, raw_image, demo)
