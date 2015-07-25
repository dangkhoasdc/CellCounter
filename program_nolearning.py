"""
File: program_nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Program version 2
"""
import cv2
import numpy as np
import sys
from skimage.feature import local_binary_pattern as lbp
from cellcounting.stage import Stage
from cellcounting.db import allidb
from cellcounting.preprocessing import morph
from cellcounting.segmentation import contour as cont
from cellcounting.fw import nolearning
from cellcounting.features.feature import Feature
from cellcounting import common as com

class GaussianAndOpening(Stage):
    """ gaussian filter + opening """
    def __init__(self, wd_sz=3):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 3}
        super(GaussianAndOpening,
              self).__init__("Gaussian and Opening operation", params)

    def run(self, image):
        gaussian_sz = (self.params["wd_sz"], self.params["wd_sz"])
        inp = image
        # com.drawHist(image, 1)
        assert inp.size > 0
        im = cv2.split(inp)[2]
        can = cv2.bilateralFilter(im, 7, 10, 60)
        thres = cv2.Canny(can, 20, 180, L2gradient=True)
        # thres = cv2.Canny(can, 10, 200)
        kernel_dilation = np.ones((3, 3), dtype=np.int8)
        kernel_erosion = np.ones((2, 2), dtype=np.uint8)
        thres = morph.dilate(thres, kernel_dilation, 4)
        thres = morph.erode(thres, kernel_erosion, 4)
        thres = morph.thinning(thres)
        com.debug_im(can)
        com.debug_im(image)
        com.debug_im(thres)
        return thres


class SegmentStage(Stage):
    """ Segmentation algorithm """
    def __init__(self, wd_sz=None):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 10, "dist_tol": 15}
        super(SegmentStage, self).__init__("findContours algorithm", params)

    def inside(self, l, s):
        """ Check if s is inside l """
        return (( l.lt[0] < s.lt[0] and l.lt[1] < s.lt[1]) and (s.rb[0] <= l.rb[0] and s.rb[1] <= l.rb[1])) \
            or (( l.lt[0] <= s.lt[0] and l.lt[1] <= s.lt[1]) and (s.rb[0] < l.rb[0] and s.rb[1] < l.rb[1]))

    def run(self, image, orig_image=None):
        dist_tol = self.params["dist_tol"]
        wd_sz = self.params["wd_sz"]
        contours = cont.findContours(image)
        contours = [con for con in contours if con.width > wd_sz and con.height > wd_sz]
        filtered_contours = []

        for con in contours:
            result = any([self.inside(con, s) for s in contours if s != con])
            if not result:
                filtered_contours.append(con)

        for con in filtered_contours:
            for c in filtered_contours:
                if con != c and com.euclid(c.center, con.center) < dist_tol:
                    filtered_contours.remove(c)


        contours = filtered_contours
        if orig_image is not None:
            for con in contours:
                con.draw(orig_image, (128, 0, 0), 1)

            com.debug_im(orig_image, False)
        return contours


if __name__ == '__main__':
    print "Main Program"
    scale = 1 / 6.0
    pre = GaussianAndOpening()
    seg = SegmentStage(15)
    framework = nolearning.NoLearningFramework(scale, pre, seg)

    try:
        ftrain = sys.argv[1]
    except:
        ftrain = "training.txt"

    file_train = open(ftrain, "r")
    row_lst = [line.strip() for line in file_train.readlines()]
    file_train.close()

    image_lst = [line+".jpg" for line in row_lst]
    loc_lst = [allidb.get_true_locs(line+".xyc", scale) for line in row_lst]
    for image, loc in zip(image_lst, loc_lst):
        framework.run(image, loc)

