"""
File: absframework.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Abstract Framrwork Class
"""
import cv2
from .. import common as com

class AbsFramework(object):
    """ The Abstract Framrwork Class """
    def __init__(self, database, preprocessing, segmentation):
        self._segmentation = segmentation
        self._preprocess = preprocessing
        self._db = database

    def preprocess(self, image):
        """ reprocessing stage """
        return self._preprocess.run(image)

    def segment(self, image):
        """ segment an image """
        return self._segmentation.run(image)

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

    def eval_segments(self, segments, loc_list):
        """
        Check if the detected segment in the list of segments
        is correctly located or not based on the list of ground true data
        Return:
            segments (list of Contour): Segments changed
            the detected attribute.
            If seg.detected == False: false counting otherwise
            seg.detected == True
        """
        for seg in segments:
            if not loc_list:
                break
            point, value = com.nearest_point(seg.center, loc_list)
            if value <= self._db.tol:
                loc_list.remove(point)
                seg.detected = True
        return segments


