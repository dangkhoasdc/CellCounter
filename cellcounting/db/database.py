"""
File: database.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The abstract class for database
"""


class Database(object):
    """ The abstract class for database
        Attributes:
            orig_size (tuple): the size of image in the database
            scale_ratio:
            radius: the approximate radius of each cell
            tol: the distance between two centers of cells
    """
    def __init__(self, name, scale_ratio, radius, tol):
        self.name = name
        self.scale_ratio = scale_ratio
        self.radius = radius
        self.tol = tol

    def __str__(self):
        return self.name

