"""
File: lbp.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Local Binary Pattern features
"""
from cellcounting.features.feature import Feature
from skimage.feature import local_binary_pattern


class LBP(Feature):
    def __init__(self, radius=2, neighbor=4, method="default"):
        super(LBP, self).__init__()
        self.r = radius
        self.p = 4
        self.method = method

    def compute(self, image):
        """
        Compute the LBP feature
        """
        return local_binary_pattern(image, self.p, self.r, self.method)

