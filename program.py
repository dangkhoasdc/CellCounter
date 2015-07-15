"""
File: program.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The main program of automatic counting of cells
"""
import cv2
import numpy as np
import sys
import common as com
from db import allidb
from preprocessing import morph


def preprocessing(img):
    """
    This is the preprocessing stage in the framework.
    It will remove noise, enhance the quality of
    an input image.

    Args:
        img (ndarray): an input image
    Return:
        img (ndarray): the result image
    """
    yield


def count_cells(image, viz=False, corpoints=None):
    """ Count of cells in an input image
        Args:
            image (str): file path of an input image
        Return:
            num (int): the number of cells in the image
    """

    inp = cv2.imread(image, 1)
    com.drawHist(image, 1)
    com.drawHist(image, 3)
    assert inp.size > 0
    inp = cv2.resize(inp, (432, 324))
    im = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
    im = cv2.GaussianBlur(im, (3, 3), 2)
    _, thres = cv2.threshold(im, 0, 255,
                             cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    # thres = cv2.adaptiveThreshold(im, 255,
    # cv2.ADAPTIVE_THRESH_MEAN_C,
    # cv2.THRESH_BINARY_INV,
    # 31, 0)
    kernel_erosion = np.ones((5, 5), dtype=np.int8)
    kernel_dilation = np.ones((4, 4), dtype=np.int8)
    thres = morph.opening(thres, kernel_erosion, kernel_dilation)
    thres = ~thres
    circles = cv2.HoughCircles(thres,
                               cv2.cv.CV_HOUGH_GRADIENT,
                               2, 12,
                               param1=50,
                               param2=30,
                               minRadius=5,
                               maxRadius=20)
    assert circles is not None
    circles = np.uint16(np.around(circles))
    if not viz:
        return circles.shape[1]

    detected_points = [(i[0], i[1]) for i in circles[0, :]]
    cv2.imshow("Original", inp)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(inp, (i[0], i[1]), 10, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(inp, (i[0], i[1]), 2, (0, 0, 255), 3)
    # show ground true data

    if corpoints:
        allidb.visualize_loc(image, corpoints)
        corpoints = [(allidb.ratio * x, allidb.ratio * y)
                     for (x, y) in corpoints]
        cordect_pts = 0
        for p in corpoints:
            cands = [cir for cir in detected_points
                     if com.euclid(p, cir) < allidb.tol]
            if len(cands) == 0:
                continue
            cordect_pts += 1
            cands = sorted(cands, lambda x, p2=x: com.euclid(p2, x))
            detected_points.remove(cands[0])
        print "The number of true cells: ", len(corpoints)
        print "The number of correctly detected cells: ", cordect_pts
        print "Error: ", len(corpoints) - cordect_pts

    cv2.imshow('detected circles', inp)
    cv2.imshow("After", thres)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return circles.shape[1]


def help_manual():
    """ manual """
    print "Automated Cell Counting Algorithm\n",\
        "Usage: \n", \
        "python program.py image\n",\
        "python program.py image ground-truth-data"

if __name__ == '__main__':
    points = None

    if sys.argv[1] in ["-h", "help"]:
        help_manual()
        sys.exit()

    if sys.argv[2]:
        f = open(sys.argv[2], "r")
        points = [tuple(map(int, loc.split())) for loc in f.readlines()]
    result = count_cells(sys.argv[1], True, points)
    print "MAIN PROGRAM"
    print "The number of detected cells: ", result
