"""
File: allidb.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Helper function of All-IDB database
"""
import cv2


def visualize_loc(img, points):
    """
    Visualize the detected ALL cells based on their location
    """
    assert isinstance(img, basestring)
    assert points is not []
    ratio = 0.25
    im = cv2.imread(img, 1)
    w, h = int(ratio * im.shape[1]), int(ratio * im.shape[0])
    im = cv2.resize(im, (w, h))
    points = [(int(ratio * x), int(ratio * y)) for x, y in points]
    for p in points:
        cv2.circle(im, p, 5, (255, 0, 0), -1)
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Display", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys

    print "Visualize the detected cells in an image"
    print "Usage:"
    print "python allidb.py image filelist"
    f = open(sys.argv[2], "r")
    points = [tuple(map(int, loc.split())) for loc in f.readlines()]

    visualize_loc(sys.argv[1], points)
