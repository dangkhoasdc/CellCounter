"""
File: absframework.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Abstract Framrwork Class
"""
import cv2

class AbsFramework(object):
    """ The Abstract Framrwork Class """
    def __init__(self, database, preprocessing, segmentation):
        self._segmentation = segmentation
        self._preprocess = preprocessing
        self._db = database

    def preprocess(self, image):
        """ reprocessing stage """
        return self._preprocess.run(image)

    def segment(self, image, raw_image, demo):
        """ segment an image """
        return self._segmentation.run(image, raw_image, demo)

    def imread(self, fname, flags=1):
        """ load an image and scale it"""

        im = cv2.imread(fname, flags)
        if im is None:
            raise IOError("Could not load an image ", fname)
        h, w = im.shape[:2]
        w = int(self._db.scale_ratio * w)
        h = int(self._db.scale_ratio * h)
        im = cv2.resize(im, (w, h))
        return im
