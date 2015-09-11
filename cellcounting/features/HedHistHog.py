"""
File: HedHistHog.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Feature: The Histogram of the HED color space + HOG
"""

import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.color import rgb2hed
from skimage.restoration import denoise_bilateral
from cellcounting import common as com
from cellcounting.features.hog import HOGFeature
from cellcounting.features.feature import Feature
from hed_bilateral import HedBilateralFilter


class HedHistHog(Feature):
    """
    HedHistHog Feature
    """
    def __init__(self):
        ###################################################################
        # Configuration
        self.image_size = (256, 256)
        self.filter_kernel = (7, 7)
        # Create samples and its label
        self.preprocessor = HedBilateralFilter()
        self.preprocessor.set_param("bilateral_kernel", self.filter_kernel)
        self.preprocessor.set_param("sigma_color", 9)
        self.hog = HOGFeature()

    def compute(self, image):
        """
        load all data form the file of list data
        """
        ########################################
        # Construct the histogram of the HED color
        hed = cv2.split(rgb2hed(image))[1]
        hed = denoise_bilateral(hed, sigma_range=0.1, sigma_spatial=1.5)
        hed[hed>=1.0] = 1.0
        hed[hed<=-1.0] = -1.0
        hed = img_as_ubyte(hed)
        hist_data = cv2.calcHist([hed], [0], None, [256], [0, 256]).flatten()

        ########################################
        # Construct the HOG feature of the image
        _, region = self.preprocessor.run(image)
        h, w = region.shape[:2]
        image_size = max(h, w)
        image_size = (image_size, image_size)
        region = cv2.resize(region, self.image_size)
        com.debug_im(region)
        hog_data, _ = self.hog.compute(region)


        hog_data = np.array(hog_data, dtype=np.float32)
        hist_data = np.array(hist_data, dtype=np.float32)
        return hist_data, hog_data
