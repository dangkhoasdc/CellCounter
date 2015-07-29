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
from skimage.morphology import watershed as ws
from skimage.morphology import erosion, disk
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)


def watershed(image):
    """ the watershed algorithm """
    if len(image.shape) != 2:
        raise TypeError("The input image must be gray-scale ")

    # thresholding
    _, thres = cv2.threshold(image, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thres = erosion(thres, disk(3))
    com.debug_im(thres)
    distance = ndi.distance_transform_edt(thres)
    local_maxi = peak_local_max(distance, indices=False,
                                footprint=np.ones((2, 2)),
                                labels=thres)

    implt = plt.imshow(-local_maxi, cmap=plt.cm.jet, interpolation='nearest')
    plt.show()
    markers = ndi.label(local_maxi)[0]
    print markers

    labels = ws(-distance, markers, mask=thres)
    labels = np.uint8(labels)
    contours, _ = cv2.findContours(labels,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    segments = [Contour(points_lst) for points_lst in contours]
    for s in segments:
        s.draw(image, (255, 255, 0), 1)
    com.debug_im(image)
    return segments
