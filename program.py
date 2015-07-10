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

def count_cells(image, viz=False):
    """ Count of cells in an input image
        Args:
            image (str): file path of an input image
        Return:
            num (int): the number of cells in the image
    """

    im =  cv2.imread(image, 0)
    assert im.size > 0
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
        cimg = cv2.imread(image)
        cv2.imshow("Original", im)
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        cv2.imshow('detected circles',cimg)
        cv2.imshow("After", thres)
        cv2.waitKey(0)
    return circles.shape[1]

if __name__ == '__main__':
    print "MAIN PROGRAM"
    print "The number of detected cells: ", count_cells(sys.argv[1], True)
