"""
File: feature.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The Abstract Feature Class
"""


class Feature(object):
    """ The Abstract Feature Class """

    def __len__(self):
        raise NotImplementedError("Subclass should be implement this ")


