"""
File: hed_bilateral.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Preprocessing stage: HED + Bilateral filter
"""
import cv2
import numpy as np
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


class HedBilateralFilter(Stage):
    """ HED color space + bilateral filter  """
    def __init__(self, wd_sz=3):
        params = {"wd_sz": wd_sz}
        self._default_params = {"wd_sz": 3}
        super(HedBilateralFilter,
              self).__init__("Hed Color space and Bilateral Filter", params)

    def run(self, image):
        assert image.size > 0
        im = img_as_ubyte(1.0 - cv2.split(rgb2hed(image))[1])
        im = rescale_intensity(im)
        im[im >= 120] = 255
        im[im <= 90] = 0
        im = rank.enhance_contrast(im, disk(5))
        im = morph.close(im, disk(3))
        can = cv2.adaptiveBilateralFilter(im,
                                          self.params["bilateral_kernel"],
                                          self.params["sigma_color"])
        thres = cv2.Canny(can, 0, 300)
        kernel_dilation = np.ones((1, 1), dtype=np.int8)
        thres = morph.dilate(thres, kernel_dilation, 2)
        return thres, can
