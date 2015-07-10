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
    assert isinstance(img,basestring)
    assert points is not []
    im = cv2.imread(img,1)
    for p in points:
        cv2.circle(im, p, 20, (1,1,0))
    cv2.imshow("Display", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys

    print "Visualize the detected cells in an image"
    print "Usage:"
    print "python allidb.py image filelist"
    f = open(sys.argv[2], "r")
    points = [map(int, loc.split()) for loc in f.readlines()]

    visualize_loc(img, points)
