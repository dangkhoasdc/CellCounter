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
from cellcounting.preprocessing import morph
from cellcounting import common as com
from skimage.morphology import disk
import skimage.filters.rank as rank
from skimage.color import rgb2hed
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity


class HedBilateralFilter(Stage):
    """ HED color space + bilateral filter  """

    def __init__(self, bilateral_kernel, sigma_color):
        self.sigma_color = sigma_color
        self.bilateral_kernel = bilateral_kernel

    def run(self, image):
        assert image.size > 0
        hed = cv2.split(rgb2hed(image))[1]
        hed = img_as_ubyte(1.0 - hed)
        # hed = 1.0 - hed
        hed = rescale_intensity(hed)
        im = hed
        # im = img_as_ubyte(hed)
        # com.debug_im(im)
        im[im >= 115] = 255
        im[im < 115] = 0
        im = rank.enhance_contrast(im, disk(5))
        im = morph.close(im, disk(3))

        can = cv2.adaptiveBilateralFilter(im,
                                          self.bilateral_kernel,
                                          self.sigma_color)
        return can
