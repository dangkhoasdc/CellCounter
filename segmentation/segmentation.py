"""
File: segmentation.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Abstract Segmentation Class
"""

class Segmentation(object):
    """ Abstract Segmentation Class"""
    def __init__(self, name):
        self.name = name

    def run(self, image):
        """Virtual Method """
        raise NotImplementedError("Subclass should implement this")


