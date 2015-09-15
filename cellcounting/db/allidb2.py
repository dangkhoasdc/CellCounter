"""
File: allidb2.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: ALL-IDB 2 database
"""
import cv2
from cellcounting.db.database import Database

resize_width, resize_height = 432, 324


class AllIdb(Database):
    """ The All-IDB database """
    def __init__(self):
        self.name = "ALL-IDB 2"
        self.scale_ratio = 1.0
        self.tol = 7
        self.radius = 8
        super(AllIdb, self).__init__(self.name,
                                     self.scale_ratio,
                                     self.radius,
                                     self.tol)


def visualize_loc(img, points, wait=False):
    """
    Visualize the detected ALL cells based on their location
    """
    assert isinstance(img, basestring)
    assert points is not []
    im = cv2.imread(img, 1)
    im = cv2.resize(im, (resize_width, resize_height))
    for p in points:
        cv2.circle(im, p, 5, (255, 0, 0), -1)
    cv2.imshow("Ground True Data", im)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
