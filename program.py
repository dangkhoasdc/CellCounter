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
from db import allidb
from segmentation import morph

def preprocessing(img):
    """
    This is the preprocessing stage in the framework. It will remove noise, enhance the quality of
    an input image.

    Args:
        img (ndarray): an input image
    Return:
        img (ndarray): the result image
    """
    yield

def count_cells(image, viz=False, points=None):
    """ Count of cells in an input image
        Args:
            image (str): file path of an input image
        Return:
            num (int): the number of cells in the image
    """

    inp =  cv2.imread(image, 1)
    assert inp.size > 0
    inp = cv2.resize(inp, (432, 324))
    im = cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY)
    im = cv2.GaussianBlur(im, (3,3), 0)
    # kernel = np.ones((5,5), dtype=np.int8)
    # proc_im = morph.opening(im, kernel)
    proc_im = im
    _, thres = cv2.threshold(proc_im, 0, 255,
                             cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(thres,cv2.cv.CV_HOUGH_GRADIENT,2,12,
                               param1=50,param2=30,minRadius=5,maxRadius=20)
    assert circles is not None
    circles = np.uint16(np.around(circles))

    if viz:
        cv2.imshow("Original", inp)
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(inp,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(inp,(i[0],i[1]),2,(0,0,255),3)
        # show ground true data

        if points:
            allidb.visualize_loc(image, points)

        cv2.imshow('detected circles',inp)
        cv2.imshow("After", thres)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return circles.shape[1]

def help_manual():
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

