"""
File: stage.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The Stage Class
"""


class Stage(object):
    """ Stage Class """

    def run(self, image):
        """ run preprocessing stage """
        raise NotImplementedError("Subclass should be implement this")
