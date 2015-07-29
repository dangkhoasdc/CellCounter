"""
File: watershed.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The watershed algorithm
"""
import numpy as np
import cv2
from cellcounting.segmentation.contour import Contour
import cellcounting.common as com
from scipy.ndimage import label


def watershed(image):
    """ the watershed algorithm """
    if len(image.shape) != 2:
        raise TypeError("The input image must be gray-scale ")

    # thresholding
    _, thres = cv2.threshold(image, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=1)
    bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(thres, cv2.cv.CV_DIST_L2, 3)
    com.debug_im(dist_transform)
    _, fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    fg = np.uint8(fg)

    unknown = cv2.subtract(bg, fg)
    markers, _ = label(fg)

    markers = markers + 1

    markers[unknown == 255] = 0
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.watershed(image, markers)
    markers[markers == -1] = 0
    markers = markers.astype(np.uint8)

    contours, _ = cv2.findContours(markers,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    segments = [Contour(points_lst) for points_lst in contours]
    for s in segments:
        s.draw(image, (255, 255, 0), 1)
    com.debug_im(image)
    return segments
