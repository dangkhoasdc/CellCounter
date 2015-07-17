"""
File: program.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The main program of automatic counting of cells
"""
import cv2
import numpy as np
import sys
from cellcounting.stage import Stage
from cellcounting.db import allidb
from cellcounting.preprocessing import morph
from cellcounting.fw import fw
from cellcounting import common as com


class GaussianAndOpening(Stage):
    """ gaussian filter + opening """
    def __init__(self, wd_sz=5):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 7}
        super(GaussianAndOpening, self).__init__("Gaussian and Opening operation", params)

    def run(self, image):
        gaussian_sz = (self.params["wd_sz"], self.params["wd_sz"])
        inp = image
        # com.drawHist(image, 1)
        assert inp.size > 0
        inp = cv2.resize(inp, (432, 324))
        im = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
        im = cv2.GaussianBlur(im, gaussian_sz, 2)
        _, thres = cv2.threshold(im, 0, 255,
                                cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # thres = cv2.adaptiveThreshold(im, 255,
        # cv2.ADAPTIVE_THRESH_MEAN_C,
        # cv2.THRESH_BINARY_INV,
        # 31, 0)
        kernel_erosion = np.ones((5, 5), dtype=np.int8)
        kernel_dilation = np.ones((4, 4), dtype=np.int8)
        thres = morph.opening(thres, kernel_erosion, kernel_dilation)
        thres = ~thres
        return thres

if __name__ == '__main__':
    print "Main Program"
    pre = GaussianAndOpening()
    image = cv2.imread(sys.argv[1])
    result = pre.run(image)
    cv2.imshow("Display", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
