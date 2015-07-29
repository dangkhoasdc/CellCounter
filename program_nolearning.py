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
from cellcounting.stage import Stage
from cellcounting.db import allidb
from cellcounting.preprocessing import morph
from cellcounting.segmentation import contour as cont
from cellcounting.fw import nolearning
from cellcounting import common as com
from skimage.morphology import disk
import skimage.filters.rank as rank
from skimage.color import rgb2hed
from skimage.restoration import denoise_tv_bregman
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

class GaussianAndOpening(Stage):
    """ gaussian filter + opening """
    def __init__(self, wd_sz=3):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 3}
        super(GaussianAndOpening,
              self).__init__("Gaussian and Opening operation", params)

    def run(self, image, visualize):
        assert image.size > 0
        im = img_as_ubyte(1.0 - cv2.split(rgb2hed(image))[1])
        can = cv2.adaptiveBilateralFilter(im,
                                          self.params["bilateral_kernel"],
                                          self.params["sigma_color"])
        can = rescale_intensity(can)
        thres = cv2.Canny(can, 0, 300)
        kernel_dilation = np.ones((2, 2), dtype=np.int8)
        kernel_erosion = np.ones((1, 1), dtype=np.uint8)
        thres = morph.dilate(thres, kernel_dilation, 4)
        thres = morph.erode(thres, kernel_erosion, 2)
        thres = morph.thinning(thres,theta=15)
        # thres = morph.skelm(thres)
        if visualize:
            com.debug_im(image)
            com.debug_im(can)
            com.debug_im(thres)
        return thres, can


class SegmentStage(Stage):
    """ Segmentation algorithm """
    def __init__(self, wd_sz=None):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 10, "dist_tol": 8}
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

    def run(self, image, raw_image, orig_image, visualize):
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
        contours = self.filter_hist(contours, raw_image)
        if visualize:
            for con in contours:
                con.draw(orig_image, (128, 0, 0), 1)

            com.debug_im(orig_image, False)
        return contours


def run_program(param, param2):
    """ run the program """
    scale = 1 / 6.0
    pre = GaussianAndOpening()
    seg = SegmentStage(5)
    framework = nolearning.NoLearningFramework(scale, pre, seg)
    filter_kernel =(param, param)
    pre.set_param("bilateral_kernel", filter_kernel)
    pre.set_param("sigma_color", param2)
    try:
        ftrain = sys.argv[1]
    except:
        ftrain = "training.txt"

    file_train = open(ftrain, "r")
    row_lst = [line.strip() for line in file_train.readlines()]
    file_train.close()

    image_lst = [line+".jpg" for line in row_lst]
    loc_lst = [allidb.get_true_locs(line+".xyc", scale) for line in row_lst]
    num_correct_items = 0
    num_detected_items = 0
    num_true_items = sum([len(loc) for loc in loc_lst])

    for image, loc in zip(image_lst, loc_lst):
        print image
        correct_items, detected_items = framework.run(image, loc, visualize=True)
        num_correct_items += correct_items
        num_detected_items += detected_items
        cv2.destroyAllWindows()

    R_ir = num_correct_items / float(num_true_items)
    P_ir = num_correct_items / float(num_detected_items)
    perf_ir = 2 * (P_ir * R_ir) / (P_ir + R_ir)
    return perf_ir

if __name__ == '__main__':
    print "Main Program"
    f = open("auto_canny.csv", "w")
    # sigma_range = np.logspace(2, 10, num=15, base=2)
    # sigma_range = [int(num) if int(num) % 2 == 1 else int(num)+1 for num in sigma_range]
    # for i in sigma_range:
    result = run_program(11, 200)
    print result
    f.write(str(result) + ",")
    f.close()
