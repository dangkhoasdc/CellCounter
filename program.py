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
from skimage.feature import local_binary_pattern as lbp
from cellcounting.stage import Stage
from cellcounting.db import allidb
from cellcounting.preprocessing import morph
from cellcounting.segmentation import contour as cont
from cellcounting.fw import fw
from cellcounting.features.feature import Feature
from cellcounting import common as com
from cellcounting.classifier.svm import SVM


class GaussianAndOpening(Stage):
    """ gaussian filter + opening """
    def __init__(self, wd_sz=5):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 7}
        super(GaussianAndOpening,
              self).__init__("Gaussian and Opening operation", params)

    def run(self, image):
        gaussian_sz = (self.params["wd_sz"], self.params["wd_sz"])
        inp = image
        # com.drawHist(image, 1)
        assert inp.size > 0
        im = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
        # im = cv2.GaussianBlur(im, gaussian_sz, 2)
        _, thres = cv2.threshold(im,
                                 0,
                                 255,
                                 cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        # thres = cv2.adaptiveThreshold(im, 255,
        # cv2.ADAPTIVE_THRESH_MEAN_C,
        # cv2.THRESH_BINARY_INV,
        # 31, 0)
        kernel_erosion = np.ones((10, 10), dtype=np.int8)
        kernel_dilation = np.ones((1, 1), dtype=np.int8)
        thres = morph.opening(thres, kernel_erosion, kernel_dilation)
        thres = ~thres
        return thres


class SegmentStage(Stage):
    """ Segmentation algorithm """
    def __init__(self, wd_sz=None):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 10}
        super(SegmentStage, self).__init__("findContours algorithm", params)

    def run(self, image):
        wd_sz = self.params["wd_sz"]
        contours = cont.findContours(image)
        contours = [con for con in contours if con.width > wd_sz and con.height > wd_sz]
        return contours


class LocalBinaryPattern(Stage, Feature):
    """ Local Binary Pattern feature """
    def __init__(self, sz):
        params = {"sz": sz}
        self._default_params = {"sz": 21}
        super(LocalBinaryPattern, self).__init__("Local Binary Pattern", params)

    def __len__(self):
        return  self.params["sz"]**2

    def run(self, image):
        """ run LBP feature """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        assert len(image.shape) == 2
        return lbp(image, 2, np.pi / 4.0).flatten()

if __name__ == '__main__':
    print "Main Program"
    pre = GaussianAndOpening()
    seg = SegmentStage(100)
    local = LocalBinaryPattern(21)
    svm = SVM(k_fold=3)
    framework = fw.Framework(21, pre, seg, local, svm)


    try:
        f = sys.argv[1]
    except:
        f = "training.txt"

    file_train = open(f, "r")
    row_lst = [line.strip() for line in file_train.readlines()]

    image_lst = [line+".jpg" for line in row_lst]
    loc_lst = [allidb.get_true_locs(line+".xyc") for line in row_lst]

    framework.run_train(image_lst, loc_lst, True)
    # image = cv2.imread(sys.argv[1])
    # image = cv2.resize(image, (432, 324))
    # result = pre.run(image)
    # cv2.imshow("Display", result)
    # conts = seg.run(result)
    # for c in conts:
        # c.draw(image, (255, 0, 0), 1)
    # cv2.imshow("display", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
