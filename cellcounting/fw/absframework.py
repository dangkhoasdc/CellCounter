"""
File: absframework.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Abstract Framrwork Class
"""


class AbsFramework(object):
    """ The Abstract Framrwork Class """
    def __init__(self):
        pass

    def preprocess(self):
        """ reprocessing stage """
        raise NotImplementedError("The Subclass should implement this")

    def segment(self):
        """ reprocessing stage """
        raise NotImplementedError("The Subclass should implement this")

    def extract(self):
        """ reprocessing stage """
        raise NotImplementedError("The Subclass should implement this")

    def train(self):
        """ training stage """
        raise NotImplementedError("The Subclass should implement this")

    def test(self):
        """ testing stage """
        raise NotImplementedError("The Subclass should implement this")
