"""
File: feature.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The Abstract Feature Class
"""


class Feature(object):
    """ The Abstract Feature Class """

    def compute(self, image):
        """ compute the feature """
        raise NotImplementedError("This method should be implemented")

