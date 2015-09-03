"""
File: hog.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: HOG Features
"""

from skimage.feature import hog
from cellcounting.features.feature import Feature
import cv2


class HOGFeature(Feature):
    """
    The HOG descriptor
    """
    def __init__(self,
                 orientations=9,
                 pixels_per_cell=(10, 10),
                 cells_per_block=(2, 2),
                 viz=False,
                 normalize=False):

        super(HOGFeature, self).__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.viz = viz
        self.normalize = normalize

    def compute(self, image):
        """
        calculate the HOG descriptor
        """
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (80, 8))
        result, visualization = hog(image,
                                    self.orientations,
                                    self.pixels_per_cell,
                                    self.cells_per_block,
                                    True,
                                    self.normalize)

        return result, visualization
