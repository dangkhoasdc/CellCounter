"""
File: lbp.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Local Binary Pattern features
"""
from cellcounting.features.feature import Feature
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
import numpy as np


class LBP(Feature):
    """
    Local Binary Pattern features
    """
    def __init__(self, radius=1, neighbor=8, method="uniform"):
        super(LBP, self).__init__()
        self.r = radius
        self.p = 8 * self.r
        self.method = method

    def compute(self, image):
        """
        Compute the LBP feature
        """
        lbp_img = local_binary_pattern(image, self.p, self.r, self.method)
        hist, _ = np.histogram(lbp_img.ravel(), 256, [0, 256])
        # normalize the hist
        hist = hist.astype(float)
        hist = normalize(hist).flatten()
        print hist
        return hist, _
