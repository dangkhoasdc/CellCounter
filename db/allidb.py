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
tol = 10

def visualize_loc(img, points, wait=False):
    """
    Visualize the detected ALL cells based on their location
    """
    assert isinstance(img, basestring)
    assert points is not []
    im = cv2.imread(img, 1)
    im = cv2.resize(im, (resize_width, resize_height))
    points = [(int(ratio * x), int(ratio * y)) for x, y in points]
    for p in points:
        cv2.circle(im, p, 5, (255, 0, 0), -1)
    cv2.imshow("Ground True Data", im)
    if wait == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys

    print "Visualize the detected cells in an image"
    print "Usage:"
    print "python allidb.py image filelist"
    f = open(sys.argv[2], "r")
    points = [tuple(map(int, loc.split())) for loc in f.readlines()]

    visualize_loc(sys.argv[1], points, True)
