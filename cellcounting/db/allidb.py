"""
File: allidb.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Helper function of All-IDB database
"""
import cv2
from cellcounting.db.database import Database

resize_width, resize_height = 432, 324


class AllIdb(Database):
    """ The All-IDB database """
    def __init__(self):
        self.name = "ALL-IDB 1"
        self.scale_ratio = 1/6.0
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


def get_true_locs(fname, scale_ratio):
    """ Get a list of expected locations in xyc file """

    if fname[-4:] != ".xyc":
        fname = fname + ".xyc"

    f = open(fname, "r")
    points = [tuple(map(int, loc.split())) for loc in f.readlines()]

    points = [p for p in points if p != ()]
    if len(points) == 0:
        return []
    points = [(int(p[0] * scale_ratio),
               int(p[1] * scale_ratio)) for p in points]
    return points

def load_data(filelist, scale_ratio):
    """
    Load all data from filelist
    Return:
        image_lst (list of string): a list of image file paths
        loc_lst (list of string): a list of location lists
    """
    files = open(filelist, "r")
    row_lst = [line.strip() for line in files.readlines()]
    files.close()

    image_lst = [line+".jpg" for line in row_lst]
    loc_lst = [get_true_locs(line+".xyc", scale_ratio) for line in row_lst]

    return image_lst, loc_lst

if __name__ == '__main__':
    import sys

    print "Visualize the detected cells in an image"
    print "Usage:"
    print "python allidb.py image filelist"

    visualize_loc(sys.argv[1], get_true_locs(sys.argv[2]), True)
