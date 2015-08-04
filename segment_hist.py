"""
File: segment_hist.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Segmentation + Histogram filter
"""

import cv2
import numpy as np
from cellcounting.stage import Stage
from cellcounting.segmentation import contour as cont
from cellcounting.segmentation.watershed import watershed
from cellcounting import common as com


class SegmentStage(Stage):
    """ Segmentation algorithm """
    def __init__(self, wd_sz=None):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 20, "dist_tol": 5}
        super(SegmentStage, self).__init__("findContours algorithm", params)

    def inside(self, l, s):
        """ Check if s is inside l """
        return (( l.lt[0] < s.lt[0] and l.lt[1] < s.lt[1]) and (s.rb[0] <= l.rb[0] and s.rb[1] <= l.rb[1])) \
            or (( l.lt[0] <= s.lt[0] and l.lt[1] <= s.lt[1]) and (s.rb[0] < l.rb[0] and s.rb[1] < l.rb[1]))

    def calcHist(self, cont):
        """ calculate the histogram of an image """
        return cv2.calcHist([cont], [0], None, [256], [0, 256]).astype(int)

    def filter_hist(self, contours, image):
        """ remove segments not containing black blocks """
        result = []
        for cont in contours:
            hist = self.calcHist(cont.get_region(image))
            sum_hist = float(np.sum(hist))
            if np.sum(hist[:95])/sum_hist >= 0.2:
                result.append(cont)
        return result

    def run(self, image, raw_image, orig_image):
        dist_tol = self.params["dist_tol"]
        wd_sz = self.params["wd_sz"]
        h, w = raw_image.shape
        contours = cont.findContours(image)
        # remove too small segments
        # with large segments, apply watershed segmentation algorithm

        filtered_contours = []
        for c in contours:
            if c.width > 27 or c.height > 27:
                segments = watershed(c.get_region(raw_image))
                if len(segments) != 0:
                    for s in segments:
                        s.lt[0] += c.lt[0]
                        s.lt[1] += c.lt[1]
                        s.rb[0] += c.lt[0]
                        s.rb[1] += c.lt[1]

                    filtered_contours.extend(segments)
                else:
                    filtered_contours.append(c)
            else:
                filtered_contours.append(c)
        contours = list(set(filtered_contours))
        contours = [c for c in contours if (c.width >= wd_sz and c.height >= wd_sz) and (1.7 > (c.width/float(c.height)) >= 0.5)]
        contours = [c for c in contours if c.area >= 140]
        contours = [c for c in contours if 8 < c.center[0] and 8 < c.center[1] and c.center[0] < w-8 and c.center[1] < h-8]
        contours = [c for c in contours if 1 < c.lt[0] and 1 < c.lt[1] and c.rb[0] < w-1 and c.rb[1] < h-1]

        for con in contours:
            for c in contours:
                if con != c and com.euclid(c.center, con.center) < dist_tol:
                    contours.remove(c)
        contours = sorted(contours, key= lambda x: x.area, reverse=True)
        filtered_contours = []
        for con in contours:
            result = any([self.inside(con, s) for s in contours if s != con])
            if not result:
                filtered_contours.append(con)
        return filtered_contours
