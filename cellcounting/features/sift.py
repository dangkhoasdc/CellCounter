"""
File: sift.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The SIFT detector and descriptor
"""
import cv2
from cellcounting.features import feature


class SIFTFeature(feature.Feature):
    """
    The SIFT detector and descriptor
    """
    extractor = cv2.DescriptorExtractor_create("SIFT")
    detector = cv2.FeatureDetector_create("SIFT")

    def __init__(self):
        # initialize the SIFT feature
        pass

    def compute(self, image, kps):
        """
        compute the SIFT feature
        """
        kps, descs = self.extractor.compute(image, kps)

        if len(kps) == 0:
            return ([], None)

        return kps, descs

    def detect(self, image):
        """
        detect the key points
        """
        kps = self.detector.detect(image)
        return kps
