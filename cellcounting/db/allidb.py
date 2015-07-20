"""
File: allidb.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Helper function of All-IDB database
"""
import cv2

resize_width, resize_height = 432, 324
ratio = 1.0/6
tol = 30


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


def get_true_locs(fname, scale_ratio=ratio):
    """ Get a list of expected locations in xyc file """
    assert fname[-3:] == "xyc"
    f = open(fname, "r")
    points = [tuple(map(int, loc.split())) for loc in f.readlines()]
    points = [(int(scale_ratio*p[0]), int(scale_ratio*p[1])) for p in points]
    return points


if __name__ == '__main__':
    import sys

    print "Visualize the detected cells in an image"
    print "Usage:"
    print "python allidb.py image filelist"

    visualize_loc(sys.argv[1], get_true_locs(sys.argv[2]), True)
